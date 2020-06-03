import glob, os, argparse, re, hashlib

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

from dataloader import SpeechCommandsGoogle


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-path", type=str, default='data.nosync/speech_commands_v0.02', help='Path to Dataset')
parser.add_argument("--batch-size", type=int, default=128, help='Batch Size')
parser.add_argument("--dataloader-num-workers", type=int, default=0, help='Number Workers Dataloader')
parser.add_argument("--validation-percentage", type=int, default=10, help='Validation Set Percentage')
parser.add_argument("--testing-percentage", type=int, default=10, help='Testing Set Percentage')
parser.add_argument("--sample-rate", type=int, default=16000, help='Audio Sample Rate')
parser.add_argument("--n-mfcc", type=int, default=40, help='Number of mfc coefficients to retain')
parser.add_argument("--win-length", type=int, default=400, help='Window size in ms')
parser.add_argument("--hop-length", type=int, default=320, help='Length of hop between STFT windows')
parser.add_argument("--word-list", nargs='+', type=str, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence'], help='Keywords to be learned')
args = parser.parse_args()



data_transform = transforms.Compose([
        torchaudio.transforms.MFCC(sample_rate = args.sample_rate, n_mfcc = args.n_mfcc, melkwargs = {'win_length' : args.win_length, 'hop_length':args.hop_length})
    ])

speech_dataset_train = SpeechCommandsGoogle(args.dataset_path, 'training', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, transform=data_transform)
speech_dataset_test = SpeechCommandsGoogle(args.dataset_path, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, transform=data_transform)
speech_dataset_val = SpeechCommandsGoogle(args.dataset_path, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, transform=data_transform)


train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)


for i_batch, sample_batch in enumerate(train_dataloader):
    x_data, y_label = sample_batch
    import pdb; pdb.set_trace()




