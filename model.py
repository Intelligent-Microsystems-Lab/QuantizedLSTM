import glob, os, argparse, re, hashlib, uuid, collections, math, time

from itertools import islice
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dataloader import SpeechCommandsGoogle
from figure_scripts import plot_curves

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")

def step_d(bits):
    return 2.0 ** (bits - 1)

class bitsplitting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, n_msb):

        if bits == None:
            return x

        l1 = (2**n_msb) -1
        l2 = 0
        beta = []
        y = []

        for i in range(n_msb):
            l2 = 2**(n_msb - (i+1))
            beta.append(l2/l1)

            y.append( torch.floor( torch.round(l1*x)/l2 ) % 2)
            import pdb; pdb.set_trace()
            y[-1] = y[-1] * beta[-1]

        ctx.beta = beta

        return torch.stack(y)

    @staticmethod
    def backward(ctx, grad_output):
        import pdb; pdb.set_trace()
        return grad_output, None, None

bitsplitter_pass = bitsplitting.apply

class QuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, x_range):
        if (x is None) or (bits is None) or (bits == 0):
            return x
        
        step_d = 2.0 ** (bits - 1)

        x_scaled = x/x_range

        x01 = torch.clamp(x_scaled,-1+(1./step_d),1-(1./step_d))

        x01q =  torch.round(x01 * step_d ) / step_d

        return x01q*x_range

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

quant_pass = QuantFunc.apply

def pact_a(x, a):
    return torch.sign(x) * .5*(torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

def limit_scale(shape, factor, beta, wb):
    fan_in = shape

    limit = torch.sqrt(torch.tensor([3*factor/fan_in]))
    if (wb is None) or (wb == 0):
        return 1, limit.item()
    
    Wm = beta/step_d(torch.tensor([float(wb)]))
    scale = 2 ** round(math.log(Wm / limit, 2.0))
    scale = scale if scale > 1 else 1.0
    limit = Wm if Wm > limit else limit

    return scale, limit.item()

def w_init(fp, wb):
    if (wb is None) or (wb == 0):
        return fp

    Wm = 1.5/step_d(torch.tensor([float(wb)]))
    return Wm if Wm > fp else fp

class CustomMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, nl, scale, wb):
        noise_w = torch.randn(weight.shape, device = input.device) * weight.max() * nl
        bias_w  = torch.randn(bias.shape, device = bias.device) * bias.max() * nl

        wq = quant_pass(weight, wb, 1.)
        bq = quant_pass(bias, wb, 1.)

        ctx.save_for_backward(input, weight, bias)
        output = input.mm(wq + noise_w) + bq + bias_w
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
	        grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
        	grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
	        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight.t(), grad_bias, None, None, None

#https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py#L32
class LSTMCellQ(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, device, cy_div, cy_scale):
        super(LSTMCellQ, self).__init__()
        self.device = device
        self.wb = wb
        self.ib = ib
        self.scale1 = 1
        self.scale2 = 1
        self.abMVM = abMVM
        self.abNM  = abNM
        self.cy_div = cy_div
        self.cy_scale = cy_scale
        self.noise_level = noise_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        self.a1 = nn.Parameter(torch.tensor([128.]))
        self.a2 = nn.Parameter(torch.tensor([16.]))
        self.a3 = nn.Parameter(torch.tensor([1.]))
        self.a4 = nn.Parameter(torch.tensor([1.]))
        self.a5 = nn.Parameter(torch.tensor([1.]))
        self.a6 = nn.Parameter(torch.tensor([1.]))
        self.a7 = nn.Parameter(torch.tensor([4.]))
        self.a8 = nn.Parameter(torch.tensor([1.]))
        self.a9 = nn.Parameter(torch.tensor([4.]))
        self.a10 = nn.Parameter(torch.tensor([1.]))
        self.a11 = nn.Parameter(torch.tensor([4.]))


    def forward(self, input, state):
        hx, cx = state

        # MVM
        gates = (CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weight_ih.t(), self.bias_ih.t(), self.noise_level, self.scale2, self.wb) + CustomMM.apply(hx, self.weight_hh.t(), self.bias_hh.t(), self.noise_level, self.scale2, self.wb))

        #i, j, f, o
        i, j, f, o = gates.chunk(4, 1)
        
        # 
        forget_gate_out = quant_pass(pact_a(torch.sigmoid(f), self.a3), self.abNM, self.a3)
        input_gate_out = quant_pass(pact_a(torch.sigmoid(i), self.a4), self.abNM, self.a4)
        activation_out = quant_pass(pact_a(torch.tanh(j), self.a5), self.abNM, self.a5)
        output_gate_out = quant_pass(pact_a(torch.sigmoid(o), self.a6), self.abNM, self.a6)

        #
        gated_cell = quant_pass(pact_a(cx * forget_gate_out, self.a7), self.abNM, self.a7)
        activated_input = quant_pass(pact_a(input_gate_out * activation_out, self.a8), self.abNM, self.a8)
        new_c = quant_pass(pact_a(gated_cell + activated_input, self.a9), self.abNM, self.a9)
        activated_cell = quant_pass(pact_a(torch.tanh(new_c), self.a10), self.abNM, self.a10)
        new_h = quant_pass(pact_a(activated_cell * output_gate_out, self.a11), self.abNM, self.a11)

        return new_h, (new_h, new_c)

class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

        limit1 = 1.0 / math.sqrt(cell_args[1])
        limit2 = 1.0 / math.sqrt(cell_args[0])

        limit1 = w_init(limit1, cell_args[2])
        limit2 = w_init(limit2, cell_args[2])

        torch.nn.init.uniform_(self.cell.weight_hh, a = -limit1, b = limit1)
        torch.nn.init.uniform_(self.cell.weight_ih, a = -limit2, b = limit2)

        #http://proceedings.mlr.press/v37/jozefowicz15.pdf
        torch.nn.init.uniform_(self.cell.bias_ih, a = -0, b = 0)
        torch.nn.init.uniform_(self.cell.bias_hh, a = 1, b = 1)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]

        return torch.stack(outputs), state

class LinLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, noise_level, abMVM, ib, wb):
        super(LinLayer, self).__init__()
        self.abMVM = abMVM
        self.ib = ib
        self.wb = wb
        self.noise_level = noise_level

        self.weights = nn.Parameter(torch.randn(inp_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))


        limit = np.sqrt(6/inp_dim)
        limit = w_init(limit, wb)

        torch.nn.init.uniform_(self.weights, a = -limit, b = limit)
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)


        self.a1 = nn.Parameter(torch.tensor([4.]))
        self.a2 = nn.Parameter(torch.tensor([16.]))

    def forward(self, input):
        return quant_pass(CustomMM.apply(quant_pass(input, self.ib, True), quant_pass(self.weights/self.scale, self.wb, True), quant_pass(self.bias/self.scale, self.wb, True), self.noise_level, self.scale, self.wb), self.abMVM, True)


class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level, cy_div, cy_scale):
        super(KWS_LSTM, self).__init__()
        self.device = device
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # LSTM layer
        self.lstmBlocks1 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        self.lstmBlocks2 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        self.lstmBlocks3 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        self.lstmBlocks4 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)

        self.lstmBlocks5 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        self.lstmBlocks6 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        self.lstmBlocks7 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        self.lstmBlocks8 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)


        # final FC layer
        self.finFC1 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)
        self.finFC2 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)
        self.finFC3 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)
        self.finFC4 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)


        self.finFC5 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)
        self.finFC6 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)
        self.finFC7 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)
        self.finFC8 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state2 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state3 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state4 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))



        self.hidden_state5 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state6 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state7 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state8 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))


        # LSTM blocks
        lstm_out1, _ = self.lstmBlocks1(inputs, self.hidden_state)
        lstm_out2, _ = self.lstmBlocks2(inputs, self.hidden_state)
        lstm_out3, _ = self.lstmBlocks3(inputs, self.hidden_state)
        lstm_out4, _ = self.lstmBlocks4(inputs, self.hidden_state)

        lstm_out5, _ = self.lstmBlocks5(inputs, self.hidden_state)
        lstm_out6, _ = self.lstmBlocks6(inputs, self.hidden_state)
        lstm_out7, _ = self.lstmBlocks7(inputs, self.hidden_state)
        lstm_out8, _ = self.lstmBlocks8(inputs, self.hidden_state)


        # final FC blocks
        output1 = self.finFC1(lstm_out1[-1,:,:])
        output2 = self.finFC2(lstm_out2[-1,:,:])
        output3 = self.finFC3(lstm_out3[-1,:,:])
        output4 = self.finFC4(lstm_out4[-1,:,:])

        output5 = self.finFC5(lstm_out5[-1,:,:])
        output6 = self.finFC6(lstm_out6[-1,:,:])
        output7 = self.finFC7(lstm_out7[-1,:,:])
        output8 = self.finFC8(lstm_out8[-1,:,:])

        output = torch.cat([output1, output2, output3, output4, output5, output6, output7, output8], 1)

        return output


def pre_processing(x, y, device, mfcc_cuda):
    batch_size = x.shape[0]

    x =  mfcc_cuda(x.to(device))
    x =  x.permute(2,0,1)
    y =  y.view((-1)).to(device)


    return x,y


