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
from model import KWS_LSTM, pre_processing, quant_pass
from figure_scripts import plot_curves

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")

torch.manual_seed(80085)
np.random.seed(80085)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general config

# parser.add_argument("--dataset-path-train", type=str, default='data.nosync/speech_commands_v0.02_cough', help='Path to Dataset')
# parser.add_argument("--dataset-path-test", type=str, default='data.nosync/speech_commands_test_set_v0.02_cough', help='Path to Dataset')
parser.add_argument("--dataset-path-train", type=str, default='data.nosync/speech_commands_v0.02', help='Path to Dataset')
parser.add_argument("--dataset-path-test", type=str, default='data.nosync/speech_commands_test_set_v0.02', help='Path to Dataset')
parser.add_argument("--batch-size", type=int, default=100, help='Batch Size')
parser.add_argument("--training-steps", type=str, default='10000,10000,10000', help='Training Steps')
parser.add_argument("--learning-rate", type=str, default='0.0005,0.0001,0.00002', help='Learning Rate')
parser.add_argument("--lstm-blocks", type=int, default=0, help='How many parallel LSTM blocks') 
parser.add_argument("--fc-blocks", type=int, default=0, help='How many parallel LSTM blocks') 
parser.add_argument("--pool-method", type=str, default="avg", help='Pooling method [max/avg]') 
parser.add_argument("--hidden", type=int, default=118, help='Number of hidden LSTM units') 
parser.add_argument("--dataloader-num-workers", type=int, default=8, help='Number Workers Dataloader')
parser.add_argument("--validation-percentage", type=int, default=10, help='Validation Set Percentage')
parser.add_argument("--testing-percentage", type=int, default=10, help='Testing Set Percentage')
parser.add_argument("--sample-rate", type=int, default=16000, help='Audio Sample Rate')
parser.add_argument("--canonical-testing", type=bool, default=False, help='Whether to use the canoncial test data.')

parser.add_argument("--n-mfcc", type=int, default=10, help='Number of mfc coefficients to retain') # 40 before
parser.add_argument("--background-volume", type=float, default=.1, help='How loud the background noise should be, between 0 and 1.') 
parser.add_argument("--background-frequency", type=float, default=.8, help='How many of the training samples have background noise mixed in.') 
parser.add_argument('--silence-percentage', type=float, default=.1, help='How much of the training data should be silence.')
parser.add_argument('--unknown-percentage', type=float, default=.1, help='How much of the training data should be unknown words.')
parser.add_argument('--time-shift-ms', type=float, default=100.0, help='Range to randomly shift the training audio by in time.')

parser.add_argument("--win-length", type=int, default=640, help='Window size in ms') # 400
parser.add_argument("--hop-length", type=int, default=320, help='Length of hop between STFT windows') #320
parser.add_argument("--std-scale", type=int, default=1, help='Scaling by how many standard deviations (e.g. how many big values will be cut off: 1std = 65%, 2std = 95%), 3std=99%') # 3

parser.add_argument("--word-list", nargs='+', type=str, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
#parser.add_argument("--word-list", nargs='+', type=str, default=['stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
# parser.add_argument("--word-list", nargs='+', type=str, default=['cough', 'unknown', 'silence'], help='Keywords to be learned')
parser.add_argument("--global-beta", type=float, default=1.5, help='Globale Beta for quantization')
parser.add_argument("--init-factor", type=float, default=2, help='Init factor for quantization')

parser.add_argument("--noise-injectionT", type=float, default=0, help='Percentage of noise injected to weights')
#parser.add_argument("--noise-injectionI", type=float, default=0, help='Percentage of noise injected to weights')
parser.add_argument("--quant-actMVM", type=int, default=0, help='Bits available for MVM activations/state')
parser.add_argument("--quant-actNM", type=int, default=0, help='Bits available for non-MVM activations/state')
parser.add_argument("--quant-inp", type=int, default=0, help='Bits available for inputs')
parser.add_argument("--quant-w", type=int, default=0, help='Bits available for weights')

parser.add_argument("--cy-div", type=int, default=1, help='CY division')
parser.add_argument("--cy-scale", type=int, default=1, help='Scaling CY')

# parser.add_argument("--inp-mean", type=float, default=-1.9685, help='Input pre_processing')
# parser.add_argument("--inp-std", type=float, default=10.8398, help='Input pre_processing')

parser.add_argument("--inp-mean", type=float, default=0, help='Input pre_processing')
parser.add_argument("--inp-std", type=float, default=1, help='Input pre_processing')

args = parser.parse_args()


args.quant_actNM = args.quant_inp = args.quant_w = args.quant_actMVM

epoch_list = np.cumsum([int(x) for x in args.training_steps.split(',')])
lr_list = [float(x) for x in args.learning_rate.split(',')]



mfcc_cuda = torchaudio.transforms.MFCC(sample_rate = args.sample_rate, n_mfcc = args.n_mfcc, log_mels = True, melkwargs = {'win_length' : args.win_length, 'hop_length' : args.hop_length, 'n_fft' : args.win_length, 'pad': 0, 'f_min' : 20, 'f_max': 4000, 'n_mels' : 40}).to(device)

speech_dataset_train = SpeechCommandsGoogle(args.dataset_path_train, 'training', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, args.background_volume, args.background_frequency, args.silence_percentage, args.unknown_percentage, args.time_shift_ms)

speech_dataset_val = SpeechCommandsGoogle(args.dataset_path_train, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0.)

speech_dataset_test = SpeechCommandsGoogle(args.dataset_path_train, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0., non_canonical_test = not args.canonical_testing)

train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)

model = KWS_LSTM(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), batch_size = args.batch_size, device = device, quant_factor = args.init_factor, quant_beta = args.global_beta, wb = args.quant_w, abMVM = args.quant_actMVM, abNM = args.quant_actNM, ib = args.quant_inp, noise_level = args.noise_injectionT, blocks = args.lstm_blocks, pool_method = args.pool_method, fc_blocks = args.fc_blocks, cy_div = args.cy_div, cy_scale = args.cy_scale).to(device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[0])  

best_acc = 0
seg_count = 1
train_acc = []
val_acc = []
model_uuid = str(uuid.uuid4())


print(args)
print(model_uuid)
print("Start Training:")
print("Epoch     Train Loss  Train Acc  Vali. Acc  Time (s)")
start_time = time.time()
for e, (x_data, y_label) in enumerate(islice(train_dataloader, epoch_list[-1])):
    if e in epoch_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[seg_count]
            seg_count += 1

    # train
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda, args.std_scale, args.inp_mean, args.inp_std)

    x_data = quant_pass(x_data, 8, 128)

    output = model(x_data)
    loss_val = loss_fn(output, y_label)
    train_acc.append((output.argmax(dim=1) == y_label).float().mean().item())

    loss_val.backward()
    optimizer.step()
    optimizer.zero_grad()

    #print("Step {0:05d} Acc {1:.4f} Loss {1:.4f}".format(e,train_acc[-1],loss_val.item()))
    if (e%100 == 0) or (e == epoch_list[-1]-1):
        # validation
        temp_list = []
        for val_e, (x_vali, y_vali) in enumerate(validation_dataloader):
            x_data, y_label = pre_processing(x_vali, y_vali, device, mfcc_cuda, args.std_scale, args.inp_mean, args.inp_std)

            x_data = quant_pass(x_data, 8, 128)

            output = model(x_data)
            temp_list.append((output.argmax(dim=1) == y_label).float().mean().item())
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

        train_time = time.time() - start_time
        start_time = time.time()
    
        print("{0:05d}     {1:.4f}      {2:.4f}     {3:.4f}     {4:.4f}".format(e, loss_val, train_acc[-1], best_acc, train_time))
        plot_curves(train_acc, val_acc, model_uuid)


# Testing
print("Start Testing:")
checkpoint_dict = torch.load('./checkpoints/'+model_uuid+'.pkl')
model.load_state_dict(checkpoint_dict['model_dict'])
acc_aux = []

for i_batch, sample_batch in enumerate(test_dataloader):
    x_data, y_label = sample_batch
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda, args.std_scale, args.inp_mean, args.inp_std)

    x_data = quant_pass(x_data, 8, 128)

    output = model(x_data)
    acc_aux.append((output.argmax(dim=1) == y_label))

test_acc = torch.cat(acc_aux).float().mean().item()
print("Test Accuracy: {0:.4f}".format(test_acc))


#checkpoint_dict = torch.load('./checkpoints/9c8bf1f3-58e5-4527-8742-2964941cbae1.pkl')
