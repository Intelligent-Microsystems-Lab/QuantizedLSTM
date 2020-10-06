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

# def shift(x):
#     if x == 0:
#         return 1
#     return 2 ** torch.round(torch.log2(x))

# def clip(x, bits):
#     if bits == 1:
#         delta = 0.
#     else:
#         delta = 1./step_d(bits)
#     maxv = +1 - delta
#     minv = -1 + delta
#     return torch.clamp(x, float(minv), float(maxv))

# def quant(x, bits, sign):
#     if bits == 1: # BNN
#         return torch.sign(x)
#     else:
#         if sign:
#             scale = step_d(bits)
#         else:
#             scale = 2.0 ** bits
#         return torch.round(x * scale ) / scale


class QuantFunc(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, bits, x_range):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.wb = wb
        #ctx.save_for_backward(x)
        # no quantizaton, if x is None or no bits given
        if (x is None) or (bits is None) or (bits == 0):
            return x
        
        step_d = 2.0 ** (bits - 1)

        x_scaled = x/x_range

        x01 = torch.clamp(x_scaled,-1+(1./step_d),1-(1./step_d))

        x01q =  torch.round(x01 * step_d ) / step_d

        return x01q*x_range

        # new experimental approach 
        #if train:
        #    return .5 * quant(x, wb, sign) + .5 * x
        #else:

        #return quant(x, wb, sign)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        STE estimator, no quantization on the backward pass
        """
        #input, = ctx.saved_tensors
        #if (input is None) or (ctx.wb is None) or (ctx.wb == 0):
        #    return grad_output, None, None, None

        return grad_output, None, None, None

def pact_a(x, a):
    return torch.sign(x) * .5*(torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

quant_pass = QuantFunc.apply

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

# noise free weights + biases in backward pass
class CustomMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, nl, scale, wb):
        noise_w = torch.randn(weight.shape, device = input.device) * weight.max() * nl
        bias_w  = torch.randn(bias.shape, device = bias.device) * bias.max() * nl

        # quant_pass for weights here + high precision backward pass weights
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
        self.a12 = nn.Parameter(torch.tensor([4.]))


    def forward(self, input, state):
        hx, cx = state

        # MVM
        gates = (CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weight_ih.t(), self.bias_ih.t(), self.noise_level, self.scale2, self.wb) + CustomMM.apply(hx, self.weight_hh.t(), self.bias_hh.t(), self.noise_level, self.scale2, self.wb))

        # we might be able to skip this one
        gates = quant_pass(pact_a(gates, self.a2), self.abMVM, self.a2)

        #i, j, f, o
        i, j, f, o = gates.chunk(4, 1)
        
        # 
        forget_gate_out = quant_pass(pact_a(torch.sigmoid(f), self.a3), self.abNM, self.a3)
        input_gate_out = quant_pass(pact_a(torch.sigmoid(i), self.a3), self.abNM, self.a3)
        activation_out = quant_pass(pact_a(torch.tanh(j), self.a3), self.abNM, self.a3)
        output_gate_out = quant_pass(pact_a(torch.sigmoid(o), self.a3), self.abNM, self.a3)


        #
        gated_cell = quant_pass(pact_a(cx * forget_gate_out, self.a3), self.abNM, self.a3)
        activated_input = quant_pass(pact_a(input_gate_out * activation_out, self.a3), self.abNM, self.a3)
        new_c = quant_pass(pact_a(gated_cell + activated_input, self.a3), self.abNM, self.a3)
        activated_cell = quant_pass(pact_a(torch.tanh(new_c), self.a3), self.abNM, self.a3)
        new_h = quant_pass(pact_a(activated_cell * output_gate_out, self.a3), self.abNM, self.a3)

        return new_h, (new_h, new_c)

        #ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # # quantize activations -> step functions
        # ingate = quant_pass(torch.sigmoid(ingate), self.abMVM, False)
        # forgetgate = quant_pass(torch.sigmoid(forgetgate), self.abMVM, False) 
        # cellgate = quant_pass(torch.tanh(cellgate), self.abMVM, True)
        # outgate = quant_pass(torch.sigmoid(outgate), self.abMVM, False)
        
        # #quantize state / cy scale
        # cy = quant_pass( (quant_pass(forgetgate * cx, self.abNM, True) + quant_pass(ingate * cellgate, self.abNM, True)) * 1/self.cy_div, self.abNM, True)
        # hy = quant_pass(outgate * quant_pass(torch.tanh(cy * self.cy_scale), self.abNM, True), self.abNM, True)


        # hx, cx = state
        # gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
        #          torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # ingate = torch.sigmoid(ingate)
        # forgetgate = torch.sigmoid(forgetgate)
        # cellgate = torch.tanh(cellgate)
        # outgate = torch.sigmoid(outgate)

        # cy = (forgetgate * cx) + (ingate * cellgate)
        # hy = outgate * torch.tanh(cy)

        #return hy, (hy, cy)


# class LSTMLayer(nn.Module):
#     def __init__(self, cell, *cell_args):
#         super(LSTMLayer, self).__init__()
#         self.cell = cell(*cell_args)

#     #@jit.script_method
#     def forward(self, input, state):
#         # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
#         inputs = input.unbind(0)
#         outputs = []#torch.jit.annotate(List[Tensor], [])
#         for i in range(len(inputs)):
#             out, state = self.cell(inputs[i], state)
#             outputs += [out]
#         return torch.stack(outputs), state

class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

        #self.cell.scale1, limit1 = limit_scale(cell_args[1], 2, 1.5, cell_args[2])
        #self.cell.scale2, limit2 = limit_scale(cell_args[0], 2, 1.5, cell_args[2])

        limit1 = 1.0 / math.sqrt(cell_args[1])
        limit2 = 1.0 / math.sqrt(cell_args[0])

        limit1 = w_init(limit1, cell_args[2])
        limit2 = w_init(limit2, cell_args[2])

        torch.nn.init.uniform_(self.cell.weight_ih, a = -limit2, b = limit2)
        torch.nn.init.uniform_(self.cell.weight_hh, a = -limit1, b = limit1)

        #torch.nn.init.uniform_(self.cell.weight_ih, a = -np.sqrt(6/cell_args[1]), b = np.sqrt(6/cell_args[1]))
        #torch.nn.init.uniform_(self.cell.weight_hh, a = -np.sqrt(6/cell_args[0]), b = np.sqrt(6/cell_args[0]))

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

        #self.scale, limit_a = limit_scale(inp_dim, 2, 1.5, wb)

        limit = np.sqrt(6/inp_dim)
        limit = w_init(limit, wb)

        torch.nn.init.uniform_(self.weights, a = -limit, b = limit)
        #torch.nn.init.uniform_(self.weights, a = -limit_a, b = limit_a)
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)


        self.a1 = nn.Parameter(torch.tensor([4.]))
        self.a2 = nn.Parameter(torch.tensor([16.]))

    def forward(self, input):
        return quant_pass(pact_a(CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weights, self.bias, self.noise_level, 1, self.wb), self.a2), self.abMVM, self.a2)
        #return quant_pass(CustomMM.apply(quant_pass(input, self.ib, True), quant_pass(self.weights/self.scale, self.wb, True), quant_pass(self.bias/self.scale, self.wb, True), self.noise_level, self.scale, self.wb), self.abMVM, True)


class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, device, quant_factor, quant_beta, wb, abMVM, abNM, blocks, ib, noise_level, pool_method, fc_blocks, cy_div, cy_scale):
        super(KWS_LSTM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_scale = 1
        self.output_scale_hist = []
        self.n_blocks = blocks
        self.fc_blocks = fc_blocks
        self.pool_method = pool_method

        # # Pooling Layer
        # if pool_method == 'max':
        #     if self.n_blocks > fc_blocks:
        #         self.poolL = nn.MaxPool1d(kernel_size = int(np.ceil(self.n_blocks/fc_blocks)))
        #     elif self.n_blocks < fc_blocks:
        #         raise ValueError('More FC Layer than LSTM Layer')
        #     else:
        #         self.poolL = None
        #     self.poolL2 = nn.MaxPool1d(kernel_size = int(np.ceil(64*fc_blocks/100)))
        # elif pool_method == 'avg':
        #     if self.n_blocks > fc_blocks:
        #         self.poolL = nn.AvgPool1d(kernel_size = int(np.ceil(self.n_blocks/fc_blocks)))
        #     elif self.n_blocks < fc_blocks:
        #         raise ValueError('More FC Layer than LSTM Layer')
        #     else:
        #         self.poolL = None
        #     self.poolL2 = nn.AvgPool1d(kernel_size = int(np.ceil(64*fc_blocks/100)))
        # else:
        #     raise ValueError('Unknown Pooling Method')

        # # LSTM blocks
        # if blocks != 0:
        #     self.lstmBlocks = []
        #     for i in range(blocks):
        #         self.lstmBlocks.append(LSTMLayer(LSTMCell, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale))
        #     self.lstmBlocks = nn.ModuleList(self.lstmBlocks)
        # else:
        #     self.lstmBlocks = LSTMLayer(LSTMCell, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)

        # # FC blocks
        # if fc_blocks != 0:
        #     self.fcBlocks = []
        #     for i in range(self.fc_blocks):
        #         self.fcBlocks.append(LinLayer(self.hidden_dim, 64, noise_level, abMVM))
        #     self.fcBlocks = nn.ModuleList(self.fcBlocks)


        # final FC layer
        self.finFC = LinLayer(self.hidden_dim, self.output_dim, noise_level, abMVM, ib, wb)


        self.lstmBlocks = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale)
        # Testing!!!!!
        #self.lstmBlocks = torch.nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = 1, batch_first = False)
        #self.finFC = torch.nn.Linear(in_features = self.hidden_dim, out_features = self.output_dim, bias = True)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))


        # # LSTM blocks
        # if self.n_blocks != 0:
        #     lstm_out = []
        #     for i in range(self.n_blocks):
        #         temp_out, _ = self.lstmBlocks[i](inputs, self.hidden_state)
        #         lstm_out.append(temp_out)
        #         del temp_out
        #     lstm_out = torch.cat(lstm_out, 2)[-1,:,:]
        #     if self.poolL:
        #         lstm_out = self.poolL(torch.unsqueeze(lstm_out, 1))[:,0,:]
        #     lstm_out = quant_pass(lstm_out, self.ib, True)
        #     lstm_out = F.pad(lstm_out, (0, self.fc_blocks*100 - lstm_out.shape[1]))
        # else:
        #     lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # # FC blocks
        # if self.fc_blocks != 0:
        #     fc_out = []
        #     for i in range(self.fc_blocks):
        #         fc_out.append(self.fcBlocks[i](lstm_out[:,i*100:(i+1)*100]))
        #     fc_out = quant_pass(self.poolL2(torch.unsqueeze(torch.cat(fc_out,1),1))[:,0,:], self.ib, True)
        #     fc_out = F.pad(fc_out, (0, 100 - fc_out.shape[1]))
        # else:
        # 	fc_out = lstm_out[-1,:,:]

        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)
        fc_out = lstm_out[-1,:,:]

        # final FC block
        output = self.finFC(fc_out)

        return output


def pre_processing(x, y, device, mfcc_cuda):
    batch_size = x.shape[0]

    x =  mfcc_cuda(x.to(device))
    x =  x.permute(2,0,1)
    y =  y.view((-1)).to(device)


    return x,y


