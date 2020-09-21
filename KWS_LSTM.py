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


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general config

# parser.add_argument("--dataset-path-train", type=str, default='data.nosync/speech_commands_v0.02_cough', help='Path to Dataset')
# parser.add_argument("--dataset-path-test", type=str, default='data.nosync/speech_commands_test_set_v0.02_cough', help='Path to Dataset')
parser.add_argument("--dataset-path-train", type=str, default='data.nosync/speech_commands_v0.02', help='Path to Dataset')
parser.add_argument("--dataset-path-test", type=str, default='data.nosync/speech_commands_test_set_v0.02', help='Path to Dataset')
parser.add_argument("--batch-size", type=int, default=256, help='Batch Size')
parser.add_argument("--validation-size", type=int, default=3, help='Number of samples used for validation')
parser.add_argument("--validation-batch", type=int, default=8192, help='Number of samples used for validation')
parser.add_argument("--epochs", type=int, default=45000, help='Epochs')
#parser.add_argument("--CE-train", type=int, default=300, help='Epochs of Cross Entropy Training')
parser.add_argument("--lr-divide", type=int, default=15000, help='Learning Rate divide')
parser.add_argument("--lstm-blocks", type=int, default=8, help='How many parallel LSTM blocks') 
parser.add_argument("--fc-blocks", type=int, default=8, help='How many parallel LSTM blocks') 
parser.add_argument("--pool-method", type=str, default="avg", help='Pooling method [max/avg]') 
parser.add_argument("--hidden", type=int, default=100, help='Number of hidden LSTM units') 
parser.add_argument("--learning-rate", type=float, default=0.0005, help='Dropout Percentage')
parser.add_argument("--dataloader-num-workers", type=int, default=4, help='Number Workers Dataloader')
parser.add_argument("--validation-percentage", type=int, default=10, help='Validation Set Percentage')
parser.add_argument("--testing-percentage", type=int, default=10, help='Testing Set Percentage')
parser.add_argument("--sample-rate", type=int, default=16000, help='Audio Sample Rate')

#could be ramped up to 128 -> explore optimal input
parser.add_argument("--n-mfcc", type=int, default=64, help='Number of mfc coefficients to retain') # 40 before
parser.add_argument("--win-length", type=int, default=400, help='Window size in ms') # 400
parser.add_argument("--hop-length", type=int, default=320, help='Length of hop between STFT windows') #320
parser.add_argument("--std-scale", type=int, default=3, help='Scaling by how many standard deviations (e.g. how many big values will be cut off: 1std = 65%, 2std = 95%), 3std=99%')

parser.add_argument("--word-list", nargs='+', type=str, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
#parser.add_argument("--word-list", nargs='+', type=str, default=['stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
# parser.add_argument("--word-list", nargs='+', type=str, default=['cough', 'unknown', 'silence'], help='Keywords to be learned')
parser.add_argument("--global-beta", type=float, default=1.5, help='Globale Beta for quantization')
parser.add_argument("--init-factor", type=float, default=2, help='Init factor for quantization')

parser.add_argument("--noise-injectionT", type=float, default=0.1, help='Percentage of noise injected to weights')
parser.add_argument("--noise-injectionI", type=float, default=0.1, help='Percentage of noise injected to weights')
parser.add_argument("--quant-actMVM", type=int, default=6, help='Bits available for MVM activations/state')
parser.add_argument("--quant-actNM", type=int, default=8, help='Bits available for non-MVM activations/state')
parser.add_argument("--quant-inp", type=int, default=4, help='Bits available for inputs')

parser.add_argument("--cy-div", type=int, default=2, help='CY division')
parser.add_argument("--cy-scale", type=int, default=2, help='Scaling CY')
parser.add_argument("--quant-w", type=int, default=None, help='Bits available for weights')
parser.add_argument("--hp-bw", type=bool, default=False, help='High precision backward pass')

args = parser.parse_args()


def step_d(bits):
    return 2.0 ** (bits - 1)

def shift(x):
    if x == 0:
        return 1
    return 2 ** torch.round(torch.log2(x))

def clip(x, bits):
    if bits == 1:
        delta = 0.
    else:
        delta = 1./step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return torch.clamp(x, float(minv), float(maxv))

def quant(x, bits, sign):
    if bits == 1: # BNN
        return torch.sign(x)
    else:
        if sign:
            scale = step_d(bits)
        else:
            scale = 2.0 ** bits
        return torch.round(x * scale ) / scale


class QuantFunc(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, wb, sign, train):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.wb = wb
        ctx.save_for_backward(x)
        # no quantizaton, if x is None or no bits given
        if (x is None) or (wb is None) or (wb == 0):
            return x 
        
        # clipping
        if sign:
            x = clip(x, wb)
        else:
            x = torch.clamp(x, 0, 1)

        # quantization

        # new experimental approach 
        if train:
            return .5 * quant(x, wb, sign) + .5 * x
        else:
            return quant(x, wb, sign)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        STE estimator, no quantization on the backward pass
        """
        input, = ctx.saved_tensors
        if (input is None) or (ctx.wb is None) or (ctx.wb == 0):
            return grad_output, None, None, None

        return grad_output, None, None, None

quant_pass = QuantFunc.apply

def limit_scale(shape, factor, beta, wb):
    fan_in = shape

    limit = torch.sqrt(torch.tensor([3*factor/fan_in]))
    if wb is None:
        return 1, limit.item()
    
    Wm = beta/step_d(torch.tensor([float(wb)]))
    scale = 2 ** round(math.log(Wm / limit, 2.0))
    scale = scale if scale > 1 else 1.0
    limit = Wm if Wm > limit else limit

    return scale, limit.item()



# noise free weights in backward pass/ optional high precision inputs in backward pass
class CustomMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, hp_inp, nl, hp_bw):
        ctx.hp_bw = hp_bw
        noise_w = torch.randn(weight.shape, device = input.device) * weight.max() * nl

        ctx.save_for_backward(input, weight, hp_inp)
        output = input.mm(weight + noise_w)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, hp_inp = ctx.saved_tensors
        grad_input = grad_output.mm(weight.t())
        
        if ctx.hp_bw:
            # use the high precision input
            grad_weight = grad_output.t().mm(hp_inp)
        else:
            grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight.t(), None, None, None

#https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py#L32
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, hp_bw, device):
        super(LSTMCell, self).__init__()
        self.device = device
        self.hp_bw = hp_bw
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

    def forward(self, input, state, train):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx, hp_hx, hp_cx = state

        # noise injection - for bias
        # noise_ih = torch.randn(self.weight_ih.t().shape, device = self.device) * self.weight_ih.max() * self.noise_level
        # noise_hh = torch.randn(self.weight_hh.t().shape, device = self.device) * self.weight_hh.max() * self.noise_level
        noise_bias_ih = torch.randn(self.bias_ih.t().shape, device = self.device) * self.bias_ih.max() * self.noise_level
        noise_bias_hh = torch.randn(self.bias_hh.t().shape, device = self.device) * self.bias_hh.max() * self.noise_level

        gates = (CustomMM.apply(quant_pass(input, self.ib, True, train), self.weight_ih.t(), input, self.noise_level, self.hp_bw) + self.bias_ih + noise_bias_ih + CustomMM.apply(quant_pass(hx, self.ib, True, train), self.weight_hh.t(), hp_hx, self.noise_level, self.hp_bw) + self.bias_hh + noise_bias_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # quantize activations -> step functions
        ingate = quant_pass(torch.sigmoid(ingate), self.abMVM, False, train)
        forgetgate = quant_pass(torch.sigmoid(forgetgate), self.abMVM, False, train) 
        cellgate = quant_pass(torch.tanh(cellgate), self.abMVM, True, train)
        outgate = quant_pass(torch.sigmoid(outgate), self.abMVM, False, train)
        
        #quantize state / cy scale
        cy = quant_pass( (quant_pass(forgetgate * cx, self.abNM, True, train) + quant_pass(ingate * cellgate, self.abNM, True, train)) * 1/args.cy_div, self.abNM, True, train)
        hy = quant_pass(outgate * quant_pass(torch.tanh(cy * args.cy_scale), self.abNM, True, train), self.abNM, True, train)

        # high precision copy of hy and cy in backward pass
        if self.hp_bw:
            hp_ingate = torch.sigmoid(ingate)
            hp_forgetgate = torch.sigmoid(forgetgate)
            hp_cellgate = torch.tanh(cellgate)
            hp_outgate = torch.sigmoid(outgate)
            hp_cy = (hp_forgetgate * hp_cx) + (ingate * cellgate) * 1/args.cy_div
            hp_hy = outgate * torch.tanh(hp_cy * args.cy_scale)
        else:
            hp_hy = None
            hp_cy = None

        
        return hy, (hy, cy, hp_hy, hp_cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

        torch.nn.init.uniform_(self.cell.weight_ih, a = -np.sqrt(6/cell_args[1]), b = np.sqrt(6/cell_args[1]))
        torch.nn.init.uniform_(self.cell.weight_hh, a = -np.sqrt(6/cell_args[0]), b = np.sqrt(6/cell_args[0]))

        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        torch.nn.init.uniform_(self.cell.bias_ih, a = -0, b = 0)
        torch.nn.init.uniform_(self.cell.bias_hh, a = 1, b = 1)

    def forward(self, input, state, train):
        inputs = input.unbind(0)
        outputs = []

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, train)
            outputs += [out]

        return torch.stack(outputs), state

class LinLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, hp_bw, noise_level, abMVM):
        super(LinLayer, self).__init__()
        self.hp_bw = hp_bw 
        self.abMVM = abMVM
        self.noise_level = noise_level

        self.weights = nn.Parameter(torch.randn(inp_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))

        torch.nn.init.uniform_(self.weights, a = -np.sqrt(6/inp_dim), b = np.sqrt(6/inp_dim))
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)


    def forward(self, input, train):
        noise_bias_ro = torch.randn(self.bias.t().shape, device = input.device) * self.bias.max() * self.noise_level

        # what activation?
        return quant_pass(torch.tanh((CustomMM.apply(input, self.weights, None, self.noise_level, self.hp_bw) + self.bias + noise_bias_ro)), self.abMVM, True, train)


class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, device, quant_factor, quant_beta, wb, abMVM, abNM, blocks, ib, noise_level, pool_method, fc_blocks, hp_bw):
        super(KWS_LSTM, self).__init__()
        self.device = device
        self.hp_bw = hp_bw
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

        # Pooling Layer
        if pool_method == 'max':
            if self.n_blocks > fc_blocks:
                self.poolL = nn.MaxPool1d(kernel_size = int(np.ceil(self.n_blocks/fc_blocks)))
            elif self.n_blocks < fc_blocks:
                raise ValueError('More FC Layer than LSTM Layer')
            else:
                self.poolL = None
            self.poolL2 = nn.MaxPool1d(kernel_size = int(np.ceil(64*fc_blocks/100)))
        elif pool_method == 'avg':
            if self.n_blocks > fc_blocks:
                self.poolL = nn.AvgPool1d(kernel_size = int(np.ceil(self.n_blocks/fc_blocks)))
            elif self.n_blocks < fc_blocks:
                raise ValueError('More FC Layer than LSTM Layer')
            else:
                self.poolL = None
            self.poolL2 = nn.AvgPool1d(kernel_size = int(np.ceil(64*fc_blocks/100)))
        else:
            raise ValueError('Unknown Pooling Method')

        # LSTM blocks
        self.lstmBlocks = []
        for i in range(blocks):
            self.lstmBlocks.append(LSTMLayer(LSTMCell, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.hp_bw, self.device))
        self.lstmBlocks = nn.ModuleList(self.lstmBlocks)

        # FC blocks
        self.fcBlocks = []
        for i in range(self.fc_blocks):
            self.fcBlocks.append(LinLayer(self.hidden_dim, 64, hp_bw, noise_level, abMVM))
        self.fcBlocks = nn.ModuleList(self.fcBlocks)

        # final FC layer
        self.finFC = LinLayer(self.hidden_dim, self.output_dim, hp_bw, noise_level, abMVM)



    def forward(self, inputs, train):
        # init states with zero
        self.hidden_state = (torch.zeros( inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros( inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))

        # LSTM blocks
        lstm_out = []
        for i in range(self.n_blocks):
            temp_out, _ = self.lstmBlocks[i](inputs, self.hidden_state, train)
            lstm_out.append(temp_out)
            del temp_out
        lstm_out = torch.cat(lstm_out, 2)[-1,:,:]
        if self.poolL:
            lstm_out = self.poolL(torch.unsqueeze(lstm_out, 1))[:,0,:]
        lstm_out = quant_pass(lstm_out, self.ib, True, train)
        lstm_out = F.pad(lstm_out, (0, self.fc_blocks*100 - lstm_out.shape[1]))

        # FC blocks
        fc_out = []
        for i in range(self.fc_blocks):
            fc_out.append(self.fcBlocks[i](lstm_out[:,i*100:(i+1)*100], train))
        fc_out = quant_pass(self.poolL2(torch.unsqueeze(torch.cat(fc_out,1),1))[:,0,:], self.ib, True, train)
        fc_out = F.pad(fc_out, (0, 100 - fc_out.shape[1]))

        # final FC block
        output = self.finFC(fc_out, train)

        return output


def pre_processing(x, y, device, mfcc_cuda, std_scale):
    batch_size = x.shape[0]

    x =  mfcc_cuda(x.to(device))
    x -= x.reshape((batch_size, -1 )).mean(axis=1)[:, None, None]
    x /= (x.reshape((batch_size, -1 )).std(axis=1)*std_scale)[:, None, None]
    x =  x.permute(2,0,1)
    y =  y.view((-1)).to(device)

    return x,y


mfcc_cuda = torchaudio.transforms.MFCC(sample_rate = args.sample_rate, n_mfcc = args.n_mfcc, melkwargs = {'win_length' : args.win_length, 'hop_length':args.hop_length}).to(device)

speech_dataset_train = SpeechCommandsGoogle(args.dataset_path_train, 'training', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, args.epochs, device = device)
speech_dataset_val = SpeechCommandsGoogle(args.dataset_path_train, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.validation_batch, args.epochs, device = device)
speech_dataset_test = SpeechCommandsGoogle(args.dataset_path_test, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, args.epochs, device = device)


train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.validation_batch, shuffle=True, num_workers=args.dataloader_num_workers)


model = KWS_LSTM(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), batch_size = args.batch_size, device = device, quant_factor = args.init_factor, quant_beta = args.global_beta, wb = args.quant_w, abMVM = args.quant_actMVM, abNM = args.quant_actNM, ib = args.quant_inp, noise_level = args.noise_injectionT, blocks = args.lstm_blocks, pool_method = args.pool_method, fc_blocks = args.fc_blocks, hp_bw = args.hp_bw).to(device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  

best_acc = 0
train_acc = []
val_acc = []
model_uuid = str(uuid.uuid4())

print(args)
print(model_uuid)
print("Start Training:")
print("Epoch     Train Loss  Train Acc  Vali. Acc  Time (s)")
for e, (x_data, y_label) in enumerate(islice(train_dataloader, args.epochs)):
    if e%args.lr_divide == 0:
        optimizer.param_groups[-1]['lr'] /= 2

    # train
    start_time = time.time()
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda, args.std_scale)

    output = model(x_data, train = True)
    # cross entropy loss
    # if e < args.CE_train:
    #     loss_val = loss_fn(output.view(-1, output.shape[-1]), torch.tensor(y_label.tolist()*output.shape[0]).to(device))
    # # max pooling loss
    # else:
    #     loss_val = loss_fn(output[(output.max(dim=2)[0]).max(dim=0)[1], torch.tensor(range(args.batch_size)), :], y_label)
    # train_acc.append((output.max(dim=0)[0].argmax(dim =1) == y_label).float().mean().item())
    loss_val = loss_fn(output, y_label)
    train_acc.append((output.argmax(dim=1) == y_label).float().mean().item())

    loss_val.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_time = time.time() - start_time

    # validation
    temp_list = []
    for val_e, (x_vali, y_vali) in enumerate(islice(train_dataloader, args.validation_size)):
        x_data, y_label = pre_processing(x_vali, y_vali, device, mfcc_cuda, args.std_scale)

        output = model(x_data, train = False)
        temp_list.append((output.argmax(dim=1) == y_label).float().mean().item())
    #val_acc.append((output.max(dim=0)[0].argmax(dim=1) == y_label).float().mean().item())
    val_acc.append(np.mean(temp_list))

    if best_acc < val_acc[-1]:
        best_acc = val_acc[-1]
        checkpoint_dict = {
            'model_dict' : model.state_dict(), 
            'optimizer'  : optimizer.state_dict(),
            'epoch'      : e, 
            'best_vali'  : best_acc, 
            'arguments'  : args,
            'train_loss' : loss_val,
            'train_curve': train_acc,
            'val_curve'  : val_acc
        }
        torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
        del checkpoint_dict

    if e%100 == 0:
        print("{0:05d}     {1:.4f}      {2:.4f}     {3:.4f}     {4:.4f}".format(e, loss_val, train_acc[-1], best_acc, train_time))
        plot_curves(train_acc, val_acc, model_uuid)


# Testing
print("Start Testing:")
checkpoint_dict = torch.load('./checkpoints/'+model_uuid+'.pkl')
model.load_state_dict(checkpoint_dict['model_dict'])
acc_aux = []

model.noise_level = args.noise_injectionI
for i_batch, sample_batch in enumerate(test_dataloader):
    x_data, y_label = sample_batch
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda, args.std_scale)

    output = model(x_data, train = False)
    #acc_aux.append((output.max(dim=0)[0].argmax(dim=1) == y_label))
    acc_aux.append((output.argmax(dim=1) == y_label))

test_acc = torch.cat(acc_aux).float().mean().item()
print("Test Accuracy: {0:.4f}".format(test_acc))


#checkpoint_dict = torch.load('./checkpoints/9c8bf1f3-58e5-4527-8742-2964941cbae1.pkl')
