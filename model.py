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

def w_init(fp, wb):
    if (wb is None) or (wb == 0):
        return fp

    Wm = 1.5/step_d(torch.tensor([float(wb)]))
    return Wm if Wm > fp else fp

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

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        STE estimator, no quantization on the backward pass
        """

        return grad_output, None, None, None

def pact_a(x, a):
    return torch.sign(x) * .5*(torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

quant_pass = QuantFunc.apply

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
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, device, cy_div, cy_scale, train_a):
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

        self.a1 = nn.Parameter(torch.tensor([128.]), requires_grad = train_a)
        self.a2 = nn.Parameter(torch.tensor([16.]), requires_grad = train_a)
        self.a3 = nn.Parameter(torch.tensor([1.]), requires_grad = train_a)
        self.a4 = nn.Parameter(torch.tensor([1.]), requires_grad = train_a)
        self.a5 = nn.Parameter(torch.tensor([1.]), requires_grad = train_a)
        self.a6 = nn.Parameter(torch.tensor([1.]), requires_grad = train_a)
        self.a7 = nn.Parameter(torch.tensor([4.]), requires_grad = train_a)
        self.a8 = nn.Parameter(torch.tensor([1.]), requires_grad = train_a)
        self.a9 = nn.Parameter(torch.tensor([4.]), requires_grad = train_a)
        self.a10 = nn.Parameter(torch.tensor([1.]), requires_grad = train_a)
        self.a11 = nn.Parameter(torch.tensor([4.]), requires_grad = train_a)


    def forward(self, input, state):
        hx, cx = state

        # MVM
        gates = (CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weight_ih.t(), self.bias_ih.t(), self.noise_level, self.scale2, self.wb) + CustomMM.apply(hx, self.weight_hh.t(), self.bias_hh.t(), self.noise_level, self.scale2, self.wb))

        i, j, f, o = gates.chunk(4, 1)
        
        forget_gate_out = quant_pass(pact_a(torch.sigmoid(f), self.a3), self.abNM, self.a3)
        input_gate_out = quant_pass(pact_a(torch.sigmoid(i), self.a4), self.abNM, self.a4)
        activation_out = quant_pass(pact_a(torch.tanh(j), self.a5), self.abNM, self.a5)
        output_gate_out = quant_pass(pact_a(torch.sigmoid(o), self.a6), self.abNM, self.a6)

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
    def __init__(self, inp_dim, out_dim, noise_level, abMVM, ib, wb, train_a):
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


        self.a1 = nn.Parameter(torch.tensor([4.]), requires_grad = train_a)
        self.a2 = nn.Parameter(torch.tensor([16.]), requires_grad = train_a)

    def forward(self, input):
        return quant_pass(pact_a(CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weights, self.bias, self.noise_level, 1, self.wb), self.a2), self.abMVM, self.a2)


class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, device, quant_factor, quant_beta, wb, abMVM, abNM, blocks, ib, noise_level, pool_method, fc_blocks, cy_div, cy_scale, train_a):
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

        # final FC layer
        self.finFC = LinLayer(self.hidden_dim, self.output_dim, noise_level, abMVM, ib, wb, train_a = train_a)

        self.lstmBlocks = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device, cy_div, cy_scale, train_a)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
       
        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # final FC block
        output = self.finFC(lstm_out[-1,:,:])


        return output


def pre_processing(x, y, device, mfcc_cuda):
    batch_size = x.shape[0]

    x =  mfcc_cuda(x.to(device))
    x =  x.permute(2,0,1)
    y =  y.view((-1)).to(device)


    return x,y


