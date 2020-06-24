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

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-path-train", type=str, default='data.nosync/speech_commands_v0.02', help='Path to Dataset')
parser.add_argument("--dataset-path-test", type=str, default='data.nosync/speech_commands_test_set_v0.02', help='Path to Dataset')
parser.add_argument("--batch-size", type=int, default=512, help='Batch Size')
parser.add_argument("--validation-size", type=int, default=1000, help='Number of batches used for validation')
parser.add_argument("--epochs", type=int, default=20000, help='Epochs')
parser.add_argument("--lr-divide", type=int, default=10000, help='Learning Rate divide')
parser.add_argument("--hidden", type=int, default=200, help='Number of hidden LSTM units') 
parser.add_argument("--learning-rate", type=float, default=0.0005, help='Dropout Percentage')
parser.add_argument("--dataloader-num-workers", type=int, default=4, help='Number Workers Dataloader')
parser.add_argument("--validation-percentage", type=int, default=10, help='Validation Set Percentage')
parser.add_argument("--testing-percentage", type=int, default=10, help='Testing Set Percentage')
parser.add_argument("--sample-rate", type=int, default=16000, help='Audio Sample Rate')
parser.add_argument("--n-mfcc", type=int, default=40, help='Number of mfc coefficients to retain')
parser.add_argument("--win-length", type=int, default=400, help='Window size in ms')
parser.add_argument("--hop-length", type=int, default=320, help='Length of hop between STFT windows')
parser.add_argument("--word-list", nargs='+', type=str, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
parser.add_argument("--global-beta", type=float, default=1.5, help='Globale Beta for quantization')
parser.add_argument("--init-factor", type=float, default=2, help='Init factor for quantization')
parser.add_argument("--std-scale", type=int, default=2, help='Scaling by how many standard deviations (e.g. how many big values will be cut off: 1std = 65%, 2std = 95%)')

parser.add_argument("--noise-injection", type=float, default=0.1, help='Percentage of noise injected to weights')
parser.add_argument("--quant-act", type=int, default=32, help='Bits available for activations/state')
parser.add_argument("--quant-inp", type=int, default=32, help='Bits available for inputs')

parser.add_argument("--cy-scale", type=int, default=2, help='Scaling CY')

# parser.add_argument("--ab1", type=int, default=32, help='Bits available for weights')
# parser.add_argument("--ab2", type=int, default=32, help='Bits available for weights')
# parser.add_argument("--ab3", type=int, default=32, help='Bits available for weights')
# parser.add_argument("--ab4", type=int, default=32, help='Bits available for weights')
# parser.add_argument("--ab5", type=int, default=None, help='Bits available for weights')
# parser.add_argument("--ab6", type=int, default=None, help='Bits available for weights')
# parser.add_argument("--ab7", type=int, default=None, help='Bits available for weights')
# parser.add_argument("--ab8", type=int, default=None, help='Bits available for weights')
# parser.add_argument("--ab9", type=int, default=None, help='Bits available for weights')
# parser.add_argument("--ab10", type=int, default=32, help='Bits available for weights')

parser.add_argument("--quant-w", type=int, default=None, help='Bits available for weights')
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
    def forward(ctx, x, wb, sign):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        # no quantizaton, if x is None or no bits given
        if (x is None) or (wb is None):
            return x 

        # clipping
        if sign:
            x = clip(x, wb)
        else:
            x = torch.clamp(x, 0, 1)

        # quantization
        return quant(x, wb, sign)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        STE estimator, no quantization on the backward pass
        """
        return grad_output, None, None

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

#https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py#L32
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ab, noise_level, device):
        super(LSTMCell, self).__init__()
        self.device = device
        self.wb = wb
        self.ab = ab
        self.noise_level = noise_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state

        # noise injection
        noise_ih = torch.randn(self.weight_ih.t().shape, device = self.device) * self.weight_ih.max() * self.noise_level
        noise_hh = torch.randn(self.weight_hh.t().shape, device = self.device) * self.weight_hh.max() * self.noise_level
        noise_bias_ih = torch.randn(self.bias_ih.t().shape, device = self.device) * self.bias_ih.max() * self.noise_level
        noise_bias_hh = torch.randn(self.bias_hh.t().shape, device = self.device) * self.bias_hh.max() * self.noise_level


        gates = (torch.mm(input, self.weight_ih.t() + noise_ih) + self.bias_ih + noise_bias_ih + torch.mm(hx, self.weight_hh.t() + noise_hh) + self.bias_hh + noise_bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # quantize activations -> step functions
        ingate = quant_pass(torch.sigmoid(ingate), self.ab, False)
        forgetgate = quant_pass(torch.sigmoid(forgetgate), self.ab, False) 
        cellgate = quant_pass(torch.tanh(cellgate), self.ab, True)
        outgate = quant_pass(torch.sigmoid(outgate), self.ab, False)
        

        #quantize state
        cy = quant_pass( (quant_pass(forgetgate * cx, self.ab, True) + quant_pass(ingate * cellgate, self.ab, True)) * 1/args.cy_scale, self.ab, True)

        # how about we give cy some scale
        # cy = forgetgate * cx + ingate * cellgate
        hy = quant_pass(outgate * quant_pass(torch.tanh(cy * args.cy_scale), self.ab, True), self.ab, True)

        
        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []

        #####
        #self.cy_hist = np.array([])
        #self.hy_hist = np.array([])
        #####

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)

            ######
            #self.hy_hist = np.concatenate((self.hy_hist, state[0].detach().cpu().numpy().flatten()))
            #self.cy_hist = np.concatenate((self.cy_hist, state[1].detach().cpu().numpy().flatten()))
            ######
            outputs += [out]
        return torch.stack(outputs), state




class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, device, quant_factor, quant_beta, wb, ab, ib, noise_level): 
        super(KWS_LSTM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.wb = wb
        self.ab = ab
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # LSTM units
        #self.lstmL = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, bias = True)
        # custom LSTM unit
        self.lstmL = LSTMLayer(LSTMCell, self.input_dim, self.hidden_dim, self.wb, self.ab, self.noise_level, self.device)
        self.LSTMState = collections.namedtuple('LSTMState', ['hx', 'cx'])

        # The linear layer that maps from hidden state space to tag space
        self.outputL = nn.Linear(self.hidden_dim, self.output_dim)

        # init weights
        self.scale_out, self.limit_out = limit_scale(self.hidden_dim, quant_factor, quant_beta, wb)
        self.scale_hh, self.limit_hh  = limit_scale(self.input_dim, quant_factor, quant_beta, wb)
        self.scale_ih, self.limit_ih  = limit_scale(self.hidden_dim, quant_factor, quant_beta, wb)
        torch.nn.init.uniform_(self.outputL.weight, a = -self.limit_out, b = self.limit_out)
        torch.nn.init.uniform_(self.lstmL.cell.weight_ih, a = -self.limit_ih, b = self.limit_ih)
        torch.nn.init.uniform_(self.lstmL.cell.weight_hh, a = -self.limit_hh, b = self.limit_hh)
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        torch.nn.init.uniform_(self.outputL.bias, a = -0, b = 0)
        torch.nn.init.uniform_(self.lstmL.cell.bias_ih, a = -0, b = 0)
        torch.nn.init.uniform_(self.lstmL.cell.bias_hh, a = 1, b = 1)

    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros( inputs.shape[1], self.hidden_dim, device = self.device),
                      torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        # input - quantized
        q_inputs = quant_pass(inputs, self.ib, True)
        # pass throug LSTM units - quantized
        lstm_out, self.hidden_state = self.lstmL(q_inputs, self.hidden_state)
        # read out layer - quantized
        outputFC = self.outputL(lstm_out[-1,:,:]) 
        output = quant_pass(torch.sigmoid(outputFC), self.ab, False)
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
speech_dataset_val = SpeechCommandsGoogle(args.dataset_path_train, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.validation_size, args.epochs, device = device)
speech_dataset_test = SpeechCommandsGoogle(args.dataset_path_test, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, args.epochs, device = device)


train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.validation_size, shuffle=True, num_workers=args.dataloader_num_workers)

model = KWS_LSTM(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), batch_size = args.batch_size, device = device, quant_factor = args.init_factor, quant_beta = args.global_beta, wb = args.quant_w, ab = args.quant_act, ib = args.quant_inp, noise_level = args.noise_injection).to(device)
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
for e, ((x_data, y_label),(x_vali, y_vali)) in enumerate(zip(islice(train_dataloader, args.epochs), islice(validation_dataloader, args.epochs))):
    if e%10000 == 0:
        optimizer.param_groups[-1]['lr'] /= 2
    # train
    start_time = time.time()
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda, args.std_scale)

    output = model(x_data)
    loss_val = loss_fn(output, y_label)
    train_acc.append((output.argmax(dim=1) == y_label).float().mean().item())
        
    loss_val.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_time = time.time() - start_time

    # validation
    x_data, y_label = pre_processing(x_vali, y_vali, device, mfcc_cuda, args.std_scale)

    output = model(x_data)
    val_acc.append((output.argmax(dim=1) == y_label).float().mean().item())

    if best_acc < val_acc[-1]:
        best_acc = val_acc[-1]
        checkpoint_dict = {
            'model_dict' : model.state_dict(), 
            'optimizer'  : optimizer.state_dict(),
            'epoch'      : e, 
            'best_vali'  : best_acc, 
            'arguments'  : args,
            'train_loss' : loss_val
        }
        torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
        del checkpoint_dict

    if e%100 == 0:
        print("{0:05d}     {1:.4f}      {2:.4f}     {3:.4f}     {4:.4f}".format(e, loss_val, train_acc[-1], best_acc, train_time))

# Testing
print("Start Testing:")
checkpoint_dict = torch.load('./checkpoints/'+model_uuid+'.pkl')
model.load_state_dict(checkpoint_dict['model_dict'])
acc_aux = []
for i_batch, sample_batch in enumerate(test_dataloader):
    x_data, y_label = sample_batch
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda, args.std_scale)

    output = model(x_data)
    acc_aux.append((output.argmax(dim=1) == y_label))

test_acc = torch.cat(acc_aux).float().mean().item()
print("Test Accuracy: {0:.4f}".format(test_acc))




# # histogram
# import matplotlib.pyplot as plt

# plt.clf()
# n, bins, patches = plt.hist( model.lstmL.cy_hist , 50, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('CY value')
# plt.ylabel('Count')
# plt.title('Histogram of CY')
# plt.grid(True)
# plt.savefig("figures/CY.png")
# plt.close()



# plt.clf()
# n, bins, patches = plt.hist( model.lstmL.hy_hist , 50, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('HY value')
# plt.ylabel('Count')
# plt.title('Histogram of HY')
# plt.grid(True)
# plt.savefig("figures/HY.png")
# plt.close()


