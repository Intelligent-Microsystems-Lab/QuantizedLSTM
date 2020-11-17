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
import pandas as pd

from dataloader import SpeechCommandsGoogle
import model_noise_test as model_lib
from model_noise_test import KWS_LSTM_mix, KWS_LSTM_bmm, KWS_LSTM_cs, pre_processing
#import model as model_lib
#from model import KWS_LSTM_mix, KWS_LSTM_bmm, KWS_LSTM_cs, pre_processing
from figure_scripts import plot_curves

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")



parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general config
parser.add_argument("--act", type=bool, default=False, help='Act')
parser.add_argument("--weight", type=bool, default=False, help='Weight')

op_flip = parser.parse_args()

checkpoint_dict = torch.load('./checkpoints/bde83981-38a9-4ff1-9504-34b182dc99e2.pkl')

args = checkpoint_dict['arguments']
model_lib.max_w = args.max_w

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

epoch_list = np.cumsum([int(x) for x in args.training_steps.split(',')])
lr_list = [float(x) for x in args.learning_rate.split(',')]

mfcc_cuda = torchaudio.transforms.MFCC(sample_rate = args.sample_rate, n_mfcc = args.n_mfcc, log_mels = True, melkwargs = {'win_length' : args.win_length, 'hop_length' : args.hop_length, 'n_fft' : args.win_length, 'pad': 0, 'f_min' : 20, 'f_max': 4000, 'n_mels' : args.n_mfcc*4}).to(device)

speech_dataset_train = SpeechCommandsGoogle(args.dataset_path_train, 'training', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, args.background_volume, args.background_frequency, args.silence_percentage, args.unknown_percentage, args.time_shift_ms)

speech_dataset_val = SpeechCommandsGoogle(args.dataset_path_train, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0.)

if not args.canonical_testing:
    speech_dataset_test = SpeechCommandsGoogle(args.dataset_path_train, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0., non_canonical_test = not args.canonical_testing)
else:
    speech_dataset_test = SpeechCommandsGoogle(args.dataset_path_test, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0., non_canonical_test = not args.canonical_testing)

train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)

if args.method == 0:
    model = KWS_LSTM_bmm(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), device = device, wb = args.quant_w, abMVM = args.quant_actMVM, abNM = args.quant_actNM, ib = args.quant_inp, noise_level = 0, drop_p = args.drop_p, n_msb = args.n_msb, pact_a = args.pact_a, bias_r = args.rows_bias)
elif args.method == 1:
    model = KWS_LSTM_cs(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), device = device, wb = args.quant_w, abMVM = args.quant_actMVM, abNM = args.quant_actNM, ib = args.quant_inp, noise_level = 0, drop_p = args.drop_p, n_msb = args.n_msb, pact_a = args.pact_a, bias_r = args.rows_bias, w_noise = -99, act_noise = -99)
elif args.method == 2:
    args.n_msb = int(args.n_msb/args.gain_blocks)
    model = KWS_LSTM_mix(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), device = device, wb = args.quant_w, abMVM = args.quant_actMVM, abNM = args.quant_actNM, ib = args.quant_inp, noise_level = 0, drop_p = args.drop_p, n_msb = args.n_msb, pact_a = args.pact_a, bias_r = args.rows_bias, gain_blocks = args.gain_blocks)
else:
    raise Exception("Unknown method: Please use 0 for quantized LSTM blocks or 1 for bit splitting.")
model.to(device)



#w_noise_list = np.arange(0,.1, .0005).repeat(20)
w_noise_list = np.arange(0,1, .0005).repeat(20)

w_res = []
act_res = []

if op_flip.weight:
    print("Weight")
    for lvl_n in w_noise_list:
        # weight noise sensitivty
        model.load_state_dict(checkpoint_dict['model_dict'])
        model.set_noise(args.noise_injectionI)
        model.set_drop_p(0)

        model.lstmBlocks.cell.act_noise = 0
        model.finFC.act_noise = 0
        model.lstmBlocks.cell.w_noise = lvl_n
        model.finFC.w_noise = lvl_n

        acc_aux = []

        for i_batch, sample_batch in enumerate(test_dataloader):
            x_data, y_label = sample_batch
            x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda)

            output = model(x_data)
            acc_aux.append((output.argmax(dim=1) == y_label))

        test_acc = torch.cat(acc_aux).float().mean().item()
        w_res.append(test_acc)

        res_df = pd.DataFrame({'w_noise_list':w_noise_list[:len(w_res)],'w_res':w_res})
        res_df.to_csv('w_noise.csv')

        print("Test Accuracy: {0:.4f} {1:.4f}".format(test_acc, lvl_n))

if op_flip.act:
    print("Act")
    for lvl_n in w_noise_list:
        # act noise sensitivty
        model.load_state_dict(checkpoint_dict['model_dict'])
        model.set_noise(args.noise_injectionI)
        model.set_drop_p(0)

        model.lstmBlocks.cell.act_noise = lvl_n
        model.finFC.act_noise = lvl_n
        model.lstmBlocks.cell.w_noise = 0
        model.finFC.w_noise = 0

        acc_aux = []

        for i_batch, sample_batch in enumerate(test_dataloader):
            x_data, y_label = sample_batch
            x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda)

            output = model(x_data)
            acc_aux.append((output.argmax(dim=1) == y_label))

        test_acc = torch.cat(acc_aux).float().mean().item()
        act_res.append(test_acc)

        res_df = pd.DataFrame({'w_noise_list':w_noise_list[:len(act_res)],'act_res':act_res})
        res_df.to_csv('act_noise.csv')

        print("Test Accuracy: {0:.4f} {1:.4f}".format(test_acc, lvl_n))