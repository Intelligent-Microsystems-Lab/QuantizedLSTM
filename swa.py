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
parser.add_argument("--epochs", type=int, default=20000, help='Epochs')
#parser.add_argument("--CE-train", type=int, default=300, help='Epochs of Cross Entropy Training')
parser.add_argument("--lr-divide", type=int, default=10000, help='Learning Rate divide')
parser.add_argument("--lstm-blocks", type=int, default=0, help='How many parallel LSTM blocks') 
parser.add_argument("--fc-blocks", type=int, default=0, help='How many parallel LSTM blocks') 
parser.add_argument("--pool-method", type=str, default="avg", help='Pooling method [max/avg]') 
parser.add_argument("--hidden", type=int, default=118, help='Number of hidden LSTM units') 
parser.add_argument("--learning-rate", type=float, default=0.0005, help='Dropout Percentage')
parser.add_argument("--dataloader-num-workers", type=int, default=4, help='Number Workers Dataloader')
parser.add_argument("--validation-percentage", type=int, default=10, help='Validation Set Percentage')
parser.add_argument("--testing-percentage", type=int, default=10, help='Testing Set Percentage')
parser.add_argument("--sample-rate", type=int, default=16000, help='Audio Sample Rate')

#could be ramped up to 128 -> explore optimal input
parser.add_argument("--n-mfcc", type=int, default=40, help='Number of mfc coefficients to retain') # 40 before
parser.add_argument("--win-length", type=int, default=400, help='Window size in ms') # 400
parser.add_argument("--hop-length", type=int, default=320, help='Length of hop between STFT windows') #320
parser.add_argument("--std-scale", type=int, default=3, help='Scaling by how many standard deviations (e.g. how many big values will be cut off: 1std = 65%, 2std = 95%), 3std=99%')

parser.add_argument("--word-list", nargs='+', type=str, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
#parser.add_argument("--word-list", nargs='+', type=str, default=['stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
# parser.add_argument("--word-list", nargs='+', type=str, default=['cough', 'unknown', 'silence'], help='Keywords to be learned')
parser.add_argument("--global-beta", type=float, default=1.5, help='Globale Beta for quantization')
parser.add_argument("--init-factor", type=float, default=2, help='Init factor for quantization')

parser.add_argument("--noise-injectionT", type=float, default=0, help='Percentage of noise injected to weights')
parser.add_argument("--noise-injectionI", type=float, default=0, help='Percentage of noise injected to weights')
parser.add_argument("--quant-actMVM", type=int, default=8, help='Bits available for MVM activations/state')
parser.add_argument("--quant-actNM", type=int, default=8, help='Bits available for non-MVM activations/state')
parser.add_argument("--quant-inp", type=int, default=8, help='Bits available for inputs')
parser.add_argument("--quant-w", type=int, default=0, help='Bits available for weights')

parser.add_argument("--cy-div", type=int, default=2, help='CY division')
parser.add_argument("--cy-scale", type=int, default=2, help='Scaling CY')
parser.add_argument("--hp-bw", type=bool, default=False, help='High precision backward pass')

args = parser.parse_args()

checkpoint_dict = torch.load('./checkpoints/9c8bf1f3-58e5-4527-8742-2964941cbae1.pkl')