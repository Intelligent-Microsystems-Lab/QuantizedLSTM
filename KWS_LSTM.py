import glob, os, argparse, re, hashlib

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
else:
    device = torch.device("cpu")


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-path", type=str, default='data.nosync/speech_commands_v0.01', help='Path to Dataset')
parser.add_argument("--batch-size", type=int, default=128, help='Batch Size')
parser.add_argument("--epochs", type=int, default=300, help='Epochs')
parser.add_argument("--num-LSTM", type=int, default=1, help='Number is stacked LSTM layers')
parser.add_argument("--hidden", type=int, default=200, help='Number of hidden LSTM units')
parser.add_argument("--dropout", type=float, default=0, help='Dropout Percentage')
parser.add_argument("--learning-rate", type=float, default=0.0005, help='Dropout Percentage')
parser.add_argument("--dataloader-num-workers", type=int, default=0, help='Number Workers Dataloader')
parser.add_argument("--validation-percentage", type=int, default=10, help='Validation Set Percentage')
parser.add_argument("--testing-percentage", type=int, default=10, help='Testing Set Percentage')
parser.add_argument("--sample-rate", type=int, default=16000, help='Audio Sample Rate')
parser.add_argument("--n-mfcc", type=int, default=40, help='Number of mfc coefficients to retain')
parser.add_argument("--win-length", type=int, default=400, help='Window size in ms')
parser.add_argument("--hop-length", type=int, default=320, help='Length of hop between STFT windows')
parser.add_argument("--word-list", nargs='+', type=str, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence'], help='Keywords to be learned')
args = parser.parse_args()



class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_LSTM, dropout, device):
        super(KWS_LSTM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_LSTM = num_LSTM
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # LSTM units
        self.lstmL = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_LSTM, bias = True, dropout = self.dropout, batch_first = True)

        # The linear layer that maps from hidden state space to tag space
        self.outputL = nn.Linear(self.hidden_dim, self.output_dim)

        # init weights
        #self.lstmL.weights.init

    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(self.num_LSTM, inputs.shape[0], self.hidden_dim, device = self.device), torch.zeros(self.num_LSTM, inputs.shape[0], self.hidden_dim, device = self.device))
        # pass throug LSTM units
        lstm_out, self.hidden_state = self.lstmL(inputs, self.hidden_state)
        # read out layer
        output = self.outputL(lstm_out[:,-1,:])
        return output




data_transform = transforms.Compose([
        torchaudio.transforms.MFCC(sample_rate = args.sample_rate, n_mfcc = args.n_mfcc, melkwargs = {'win_length' : args.win_length, 'hop_length':args.hop_length})
    ])

speech_dataset_train = SpeechCommandsGoogle(args.dataset_path, 'training', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, device = device, transform=data_transform)
speech_dataset_test = SpeechCommandsGoogle(args.dataset_path, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, device = device, transform=data_transform)
speech_dataset_val = SpeechCommandsGoogle(args.dataset_path, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, device = device, transform=data_transform)


train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)

model = KWS_LSTM(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list)+1, batch_size = args.batch_size, num_LSTM = args.num_LSTM, dropout = args.dropout, device = device).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  


print("Start Training:")
for e in range(args.epochs):
    # train
    acc_aux = []
    for i_batch, sample_batch in enumerate(train_dataloader):
        x_data, y_label = sample_batch
        y_label = y_label.to(device)
        output = model(x_data)
        loss_val = loss_fn(output, y_label)
        acc_aux.append((output.argmax(dim=1) == y_label))
        
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()
        import pdb; pdb.set_trace()
    train_acc = torch.cat(acc_aux).float().mean().item()

    # validation
    acc_aux = []
    for i_batch, sample_batch in enumerate(validation_dataloader):
        x_data, y_label = sample_batch
        y_label = y_label.to(device)
        output = model(x_data)
        acc_aux.append((output.argmax(dim=1) == y_label))
    val_acc = torch.cat(acc_aux).float().mean().item()

    print("Epoch {0:02d}: Train Loss {1:.4f}, Train Acc {2:.4f}, Validation Acc {3:.4f}".format(e, loss_val, train_acc, val_acc))


# Testing
print("Start Testing:")
acc_aux = []
for i_batch, sample_batch in enumerate(test_dataloader):
    x_data, y_label = sample_batch
    y_label = y_label.to(device)
    output = model(x_data)
    acc_aux.append()
    acc_aux.append((output.argmax(dim=1) == y_label))

test_acc = torch.cat(acc_aux).float().mean().item()
print("Test Accuracy: {0:.4f}".format(test_acc))

