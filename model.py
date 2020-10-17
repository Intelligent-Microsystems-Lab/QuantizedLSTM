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

def pre_processing(x, y, device, mfcc_cuda):
    batch_size = x.shape[0]

    x =  mfcc_cuda(x.to(device))
    x =  x.permute(2,0,1)
    y =  y.view((-1)).to(device)

    return x,y


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


def step_d(bits):
    return 2.0 ** (bits - 1)

class QuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, x_range):
        if (x is None) or (bits is None) or (bits == 0):
            return x
        
        step_d = 2.0 ** (bits - 1)

        if len(x_range) > 1:
            x_range = x_range.unsqueeze(1).unsqueeze(1).expand(x.shape)

            x_scaled = x/x_range

            x01 = torch.clamp(x_scaled,-1+(1./step_d),1-(1./step_d))

            x01q =  torch.round(x01 * step_d ) / step_d

            x = x01q*x_range

        else:
            x_scaled = x/x_range

            x01 = torch.clamp(x_scaled,-1+(1./step_d),1-(1./step_d))

            x01q =  torch.round(x01 * step_d ) / step_d

            x = x01q*x_range

        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

quant_pass = QuantFunc.apply

class bitsplitting_sym(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, n_msb):
        if bits == None or n_msb == None:
            return x

        beta = torch.tensor([1.], requires_grad = False).to(x.device)
        y = []

        for i in range(n_msb):
            y.append(quant_pass(x/beta[-1], bits, torch.tensor([1]).to(x.device)))
            x = x - y[-1]*beta[-1]
            beta = torch.cat((beta, (beta[-1]/2.).unsqueeze(0)),0)

        return torch.stack(y).to(x.device), beta[:-1].to(x.device)

    @staticmethod
    def backward(ctx, grad_output, grad_beta):
        return grad_output.sum(0), None, None

bitsplitter_sym_pass = bitsplitting_sym.apply


def pact_a(x, a):
    return torch.sign(x) * .5*(torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

def pact_a_bmm(x, a):
    a = a.unsqueeze(1).unsqueeze(1).expand(x.shape)
    return torch.sign(x) * .5 * (torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

def w_init(fp, wb):
    if (wb is None) or (wb == 0):
        return fp

    Wm = 1.5/step_d(torch.tensor([float(wb)]))
    return Wm if Wm > fp else fp


class CustomMM_bmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, nl, wb):
        noise_w = torch.randn(weight.shape, device = input.device) * weight.max() * nl
        bias_w  = torch.randn(bias.shape, device = bias.device) * bias.max() * nl

        wq = quant_pass(weight, wb, 1.)
        bq = quant_pass(bias, wb, 1.)

        ctx.save_for_backward(input, weight, bias)
        output = input.bmm(wq + noise_w) + bq + bias_w
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.bmm(weight.permute(0,2,1))
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.permute(0,2,1).bmm(input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(1)

        return grad_input, grad_weight.permute(0,2,1), grad_bias.unsqueeze(1), None, None, None


class LSTMCellQ_bs(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, n_msb, device):
        super(LSTMCellQ_bs, self).__init__()
        self.device = device
        self.wb = wb
        self.ib = ib
        self.abMVM = abMVM
        self.abNM  = abNM
        self.noise_level = noise_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_msb = n_msb
        self.weight_ih = nn.Parameter(torch.randn(1, input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1, hidden_size, 4 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(1, 1, 4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1, 1, 4 * hidden_size))

        self.a1 = nn.Parameter(torch.tensor([128.] ))
        self.a2 = nn.Parameter(torch.tensor([16.] ))
        self.a3 = nn.Parameter(torch.tensor([1.] ))
        self.a4 = nn.Parameter(torch.tensor([1.] ))
        self.a5 = nn.Parameter(torch.tensor([1.] ))
        self.a6 = nn.Parameter(torch.tensor([1.] ))
        self.a7 = nn.Parameter(torch.tensor([4.] ))
        self.a8 = nn.Parameter(torch.tensor([1.] ))
        self.a9 = nn.Parameter(torch.tensor([4.] ))
        self.a10 = nn.Parameter(torch.tensor([1.] ))
        self.a11 = nn.Parameter(torch.tensor([4.] ))


    def forward(self, input, state):
        hx, cx = state

        inp_msb, beta_coef = bitsplitter_sym_pass(pact_a(input, self.a1)/self.a1, self.ib, self.n_msb)
        out = CustomMM_bmm.apply(inp_msb, self.weight_ih.expand(self.n_msb, self.weight_ih.shape[1], self.weight_ih.shape[2]), self.bias_ih.expand(self.n_msb, 1, self.bias_ih.shape[2]), self.noise_level, self.wb)
        out_q = quant_pass(out, self.abMVM, torch.tensor([1]).to(input.device))
        part1 = (beta_coef.unsqueeze(1).unsqueeze(1).expand(out_q.shape) * out_q).sum(0) * self.a1


        inp_msb, beta_coef = bitsplitter_sym_pass(pact_a(hx, self.a11)/self.a11, self.ib, self.n_msb)
        out = CustomMM_bmm.apply(inp_msb, self.weight_hh.expand(self.n_msb, self.weight_hh.shape[1], self.weight_hh.shape[2]), self.bias_hh.expand(self.n_msb, 1, self.bias_hh.shape[2]), self.noise_level, self.wb)
        out_q = quant_pass(out, self.abMVM, torch.tensor([1]).to(input.device))
        part2 = (beta_coef.unsqueeze(1).unsqueeze(1).expand(out_q.shape) * out_q).sum(0) * self.a11

        gates = part1 + part2
        # MVM
        #gates = (CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weight_ih, self.bias_ih, self.noise_level, self.wb) + CustomMM.apply(hx, self.weight_hh, self.bias_hh, self.noise_level, self.wb))

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


class LinLayer_bs(nn.Module):
    def __init__(self, inp_dim, out_dim, noise_level, abMVM, ib, wb, n_msb):
        super(LinLayer_bs, self).__init__()
        self.abMVM = abMVM
        self.ib = ib
        self.wb = wb
        self.n_msb = n_msb
        self.noise_level = noise_level
        self.out_dim = out_dim

        self.weights = nn.Parameter(torch.randn(1, inp_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(1, 1, out_dim))


        limit = np.sqrt(6/inp_dim)
        limit = w_init(limit, wb)

        torch.nn.init.uniform_(self.weights, a = -limit, b = limit)
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)


        self.a1 = nn.Parameter(torch.tensor([4.]))
        self.a2 = nn.Parameter(torch.tensor([16.]))

    def forward(self, input):

        inp_msb, beta_coef = bitsplitter_sym_pass(pact_a(input, self.a1)/self.a1, self.ib, self.n_msb)
        out = CustomMM_bmm.apply(inp_msb, self.weights.expand(self.n_msb, self.weights.shape[1], self.weights.shape[2]), self.bias.expand(self.n_msb, 1, self.bias.shape[2]), self.noise_level, self.wb)
        out_q = quant_pass(out, self.abMVM, torch.tensor([1]).to(input.device))
        return (beta_coef.unsqueeze(1).unsqueeze(1).expand(out_q.shape) * out_q).sum(0) * self.a1


class KWS_LSTM_bs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level, n_msb):
        super(KWS_LSTM_bs, self).__init__()
        self.device = device
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_msb = n_msb

        # LSTM layer
        self.lstmBlocks = LSTMLayer(LSTMCellQ_bs, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.n_msb, self.device)

        # final FC layer
        self.finFC = LinLayer_bs(self.hidden_dim, output_dim, noise_level, abMVM, ib, wb, n_msb)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))

        # LSTM blocks
        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # final FC blocks
        output = self.finFC(lstm_out[-1,:,:])

        return output

    def set_noise(self, nl):
        self.noise_level = nl
        self.lstmBlocks.cell.noise_level = nl
        self.finFC.noise_level = nl

    def get_a(self):
        return torch.cat([self.lstmBlocks.cell.a1, self.lstmBlocks.cell.a3, self.lstmBlocks.cell.a2,  self.lstmBlocks.cell.a4, self.lstmBlocks.cell.a5, self.lstmBlocks.cell.a6, self.lstmBlocks.cell.a7, self.lstmBlocks.cell.a8, self.lstmBlocks.cell.a9, self.lstmBlocks.cell.a10,  self.lstmBlocks.cell.a11, self.finFC.a1, self.finFC.a2])/13



class LSTMCellQ_bmm(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, n_blocks, device):
        super(LSTMCellQ_bmm, self).__init__()
        self.device = device
        self.wb = wb
        self.ib = ib
        self.abMVM = abMVM
        self.abNM  = abNM
        self.noise_level = noise_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.weight_ih = nn.Parameter(torch.randn(n_blocks, input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(n_blocks, hidden_size, 4 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(n_blocks, 1, 4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(n_blocks, 1, 4 * hidden_size))

        self.a1 = nn.Parameter(torch.tensor([128.] * n_blocks))
        self.a2 = nn.Parameter(torch.tensor([16.] * n_blocks))
        self.a3 = nn.Parameter(torch.tensor([1.] * n_blocks))
        self.a4 = nn.Parameter(torch.tensor([1.] * n_blocks))
        self.a5 = nn.Parameter(torch.tensor([1.] * n_blocks))
        self.a6 = nn.Parameter(torch.tensor([1.] * n_blocks))
        self.a7 = nn.Parameter(torch.tensor([4.] * n_blocks))
        self.a8 = nn.Parameter(torch.tensor([1.] * n_blocks))
        self.a9 = nn.Parameter(torch.tensor([4.] * n_blocks))
        self.a10 = nn.Parameter(torch.tensor([1.] * n_blocks))
        self.a11 = nn.Parameter(torch.tensor([4.] * n_blocks))


    def forward(self, input, state):
        hx, cx = state

        # MVM
        gates = (CustomMM_bmm.apply(quant_pass(pact_a_bmm(input.repeat(self.n_blocks, 1, 1), self.a1), self.ib, self.a1), self.weight_ih, self.bias_ih, self.noise_level, self.wb) + CustomMM_bmm.apply(quant_pass(pact_a_bmm(hx, self.a11), self.ib, self.a11), self.weight_hh, self.bias_hh, self.noise_level, self.wb))

        #i, j, f, o
        i, j, f, o = gates.chunk(4, 2)
        
        # 
        forget_gate_out = quant_pass(pact_a_bmm(torch.sigmoid(f), self.a3), self.abMVM, self.a3)
        input_gate_out = quant_pass(pact_a_bmm(torch.sigmoid(i), self.a4), self.abMVM, self.a4)
        activation_out = quant_pass(pact_a_bmm(torch.tanh(j), self.a5), self.abMVM, self.a5)
        output_gate_out = quant_pass(pact_a_bmm(torch.sigmoid(o), self.a6), self.abMVM, self.a6)

        #
        gated_cell = quant_pass(pact_a_bmm(cx * forget_gate_out, self.a7), self.abNM, self.a7)
        activated_input = quant_pass(pact_a_bmm(input_gate_out * activation_out, self.a8), self.abNM, self.a8)
        new_c = quant_pass(pact_a_bmm(gated_cell + activated_input, self.a9), self.abNM, self.a9)
        activated_cell = quant_pass(pact_a_bmm(torch.tanh(new_c), self.a10), self.abNM, self.a10)
        new_h = quant_pass(pact_a_bmm(activated_cell * output_gate_out, self.a11), self.abNM, self.a11)

        return new_h, (new_h, new_c)

class LinLayer_bmm(nn.Module):
    def __init__(self, inp_dim, out_dim, noise_level, abMVM, ib, wb, n_blocks):
        super(LinLayer_bmm, self).__init__()
        self.abMVM = abMVM
        self.ib = ib
        self.wb = wb
        self.noise_level = noise_level

        self.weights = nn.Parameter(torch.randn(n_blocks, inp_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(n_blocks, 1, out_dim))


        limit = np.sqrt(6/inp_dim)
        limit = w_init(limit, wb)

        torch.nn.init.uniform_(self.weights, a = -limit, b = limit)
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)


        self.a1 = nn.Parameter(torch.tensor([4.]*n_blocks))
        self.a2 = nn.Parameter(torch.tensor([16.]*n_blocks))

    def forward(self, input):
        return quant_pass(pact_a_bmm(CustomMM_bmm.apply(quant_pass(pact_a_bmm(input, self.a1), self.ib, self.a1), self.weights, self.bias, self.noise_level, self.wb), self.a2), self.abMVM, self.a2)


class KWS_LSTM_bmm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level):
        super(KWS_LSTM_bmm, self).__init__()
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
        self.lstmBlocks = LSTMLayer(LSTMCellQ_bmm, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, 8, self.device)

        # final FC layer
        self.finFC = LinLayer_bmm(self.hidden_dim, 2, noise_level, abMVM, ib, wb, 8)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(8, inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(8, inputs.shape[1], self.hidden_dim, device = self.device))
        
        # LSTM blocks
        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # final FC blocks
        output = self.finFC(lstm_out[-1,:,:,:])

        output = torch.stack([output[0,:,0] + output[0,:,1], output[1,:,0] + output[1,:,1], output[2,:,0] + output[2,:,1], output[3,:,0] + output[3,:,1], output[4,:,0], output[4,:,1], output[5,:,0], output[5,:,1], output[6,:,0], output[6,:,1], output[7,:,0], output[7,:,1]],0).t()

        return output

    def set_noise(self, nl):
        self.noise_level = nl
        self.lstmBlocks.cell.noise_level = nl
        self.finFC.noise_level = nl


    def get_a(self):
        return torch.cat([self.lstmBlocks.cell.a1, self.lstmBlocks.cell.a3, self.lstmBlocks.cell.a2,  self.lstmBlocks.cell.a4, self.lstmBlocks.cell.a5, self.lstmBlocks.cell.a6, self.lstmBlocks.cell.a7, self.lstmBlocks.cell.a8, self.lstmBlocks.cell.a9, self.lstmBlocks.cell.a10,  self.lstmBlocks.cell.a11, self.finFC.a1, self.finFC.a2])/104











class CustomMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, nl, wb):
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
        return quant_pass(pact_a(CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weights, self.bias, self.noise_level, self.wb), self.a2), self.abMVM, self.a2)




class LSTMCellQ(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, device):
        super(LSTMCellQ, self).__init__()
        self.device = device
        self.wb = wb
        self.ib = ib
        self.abMVM = abMVM
        self.abNM  = abNM
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
        gates = (CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weight_ih.t(), self.bias_ih.t(), self.noise_level, self.wb) + CustomMM.apply(hx, self.weight_hh.t(), self.bias_hh.t(), self.noise_level, self.wb))

        #i, j, f, o
        i, j, f, o = gates.chunk(4, 1)
        
        # 
        forget_gate_out = quant_pass(pact_a(torch.sigmoid(f), self.a3), self.abMVM, self.a3)
        input_gate_out = quant_pass(pact_a(torch.sigmoid(i), self.a4), self.abMVM, self.a4)
        activation_out = quant_pass(pact_a(torch.tanh(j), self.a5), self.abMVM, self.a5)
        output_gate_out = quant_pass(pact_a(torch.sigmoid(o), self.a6), self.abMVM, self.a6)

        #
        gated_cell = quant_pass(pact_a(cx * forget_gate_out, self.a7), self.abNM, self.a7)
        activated_input = quant_pass(pact_a(input_gate_out * activation_out, self.a8), self.abNM, self.a8)
        new_c = quant_pass(pact_a(gated_cell + activated_input, self.a9), self.abNM, self.a9)
        activated_cell = quant_pass(pact_a(torch.tanh(new_c), self.a10), self.abNM, self.a10)
        new_h = quant_pass(pact_a(activated_cell * output_gate_out, self.a11), self.abNM, self.a11)

        return new_h, (new_h, new_c)



class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level):
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
        self.lstmBlocks1 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks2 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks3 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks4 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)

        self.lstmBlocks5 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks6 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks7 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks8 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)


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

    def set_noise(self, nl):
        self.noise_level = nl

        self.lstmBlocks1.cell.noise_level = nl
        self.lstmBlocks2.cell.noise_level = nl
        self.lstmBlocks3.cell.noise_level = nl
        self.lstmBlocks4.cell.noise_level = nl
        self.lstmBlocks5.cell.noise_level = nl
        self.lstmBlocks6.cell.noise_level = nl
        self.lstmBlocks7.cell.noise_level = nl
        self.lstmBlocks8.cell.noise_level = nl

        self.finFC1.noise_level = nl
        self.finFC2.noise_level = nl
        self.finFC3.noise_level = nl
        self.finFC4.noise_level = nl
        self.finFC5.noise_level = nl
        self.finFC6.noise_level = nl
        self.finFC7.noise_level = nl
        self.finFC8.noise_level = nl


    def get_a(self):
        return torch.cat([self.lstmBlocks1.cell.a1, self.lstmBlocks1.cell.a3, self.lstmBlocks1.cell.a2,  self.lstmBlocks1.cell.a4, self.lstmBlocks1.cell.a5, self.lstmBlocks1.cell.a6, self.lstmBlocks1.cell.a7, self.lstmBlocks1.cell.a8, self.lstmBlocks1.cell.a9, self.lstmBlocks1.cell.a10,  self.lstmBlocks1.cell.a11, self.finFC1.a1, self.finFC1.a2, self.lstmBlocks2.cell.a1, self.lstmBlocks2.cell.a3, self.lstmBlocks2.cell.a2,  self.lstmBlocks2.cell.a4, self.lstmBlocks2.cell.a5, self.lstmBlocks2.cell.a6, self.lstmBlocks2.cell.a7, self.lstmBlocks2.cell.a8, self.lstmBlocks2.cell.a9, self.lstmBlocks2.cell.a10,  self.lstmBlocks2.cell.a11, self.finFC2.a1, self.finFC2.a2, self.lstmBlocks3.cell.a1, self.lstmBlocks3.cell.a3, self.lstmBlocks3.cell.a2,  self.lstmBlocks3.cell.a4, self.lstmBlocks3.cell.a5, self.lstmBlocks3.cell.a6, self.lstmBlocks3.cell.a7, self.lstmBlocks3.cell.a8, self.lstmBlocks3.cell.a9, self.lstmBlocks3.cell.a10,  self.lstmBlocks3.cell.a11, self.finFC3.a1, self.finFC3.a2, self.lstmBlocks4.cell.a1, self.lstmBlocks4.cell.a3, self.lstmBlocks4.cell.a2,  self.lstmBlocks4.cell.a4, self.lstmBlocks4.cell.a5, self.lstmBlocks4.cell.a6, self.lstmBlocks4.cell.a7, self.lstmBlocks4.cell.a8, self.lstmBlocks4.cell.a9, self.lstmBlocks4.cell.a10,  self.lstmBlocks4.cell.a11, self.finFC4.a1, self.finFC4.a2, self.lstmBlocks5.cell.a1, self.lstmBlocks5.cell.a3, self.lstmBlocks5.cell.a2,  self.lstmBlocks5.cell.a4, self.lstmBlocks5.cell.a5, self.lstmBlocks5.cell.a6, self.lstmBlocks5.cell.a7, self.lstmBlocks5.cell.a8, self.lstmBlocks5.cell.a9, self.lstmBlocks5.cell.a10,  self.lstmBlocks5.cell.a11, self.finFC5.a1, self.finFC5.a2, self.lstmBlocks6.cell.a1, self.lstmBlocks6.cell.a3, self.lstmBlocks6.cell.a2,  self.lstmBlocks6.cell.a4, self.lstmBlocks6.cell.a5, self.lstmBlocks6.cell.a6, self.lstmBlocks6.cell.a7, self.lstmBlocks6.cell.a8, self.lstmBlocks6.cell.a9, self.lstmBlocks6.cell.a10,  self.lstmBlocks6.cell.a11, self.finFC6.a1, self.finFC6.a2, self.lstmBlocks7.cell.a1, self.lstmBlocks7.cell.a3, self.lstmBlocks7.cell.a2,  self.lstmBlocks7.cell.a4, self.lstmBlocks7.cell.a5, self.lstmBlocks7.cell.a6, self.lstmBlocks7.cell.a7, self.lstmBlocks7.cell.a8, self.lstmBlocks7.cell.a9, self.lstmBlocks7.cell.a10,  self.lstmBlocks7.cell.a11, self.finFC7.a1, self.finFC7.a2, self.lstmBlocks8.cell.a1, self.lstmBlocks8.cell.a3, self.lstmBlocks8.cell.a2,  self.lstmBlocks8.cell.a4, self.lstmBlocks8.cell.a5, self.lstmBlocks8.cell.a6, self.lstmBlocks8.cell.a7, self.lstmBlocks8.cell.a8, self.lstmBlocks8.cell.a9, self.lstmBlocks8.cell.a10,  self.lstmBlocks8.cell.a11, self.finFC8.a1, self.finFC8.a2])/104

