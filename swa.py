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
from model import KWS_LSTM, pre_processing

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", type=str, default='50127940-556e-49d7-95a9-9bf94c182714', help='checkpoint to perfrom SWA on')
parser.add_argument("--training-cycles", type=int, default=100, help='Training Steps')
parser.add_argument("--cycle-steps", type=int, default=30, help='Training Steps')
args_swa = parser.parse_args()


checkpoint_dict = torch.load('./checkpoints/'+args_swa.checkpoint+'.pkl')

args = checkpoint_dict['arguments']
epoch_list = np.cumsum([int(x) for x in args.training_steps.split(',')])


mfcc_cuda = torchaudio.transforms.MFCC(sample_rate = args.sample_rate, n_mfcc = args.n_mfcc, log_mels = True, melkwargs = {'win_length' : args.win_length, 'hop_length' : args.hop_length, 'n_fft' : args.win_length, 'pad': 0, 'f_min' : 20, 'f_max': 4000, 'n_mels' : 40}).to(device)

speech_dataset_train = SpeechCommandsGoogle(args.dataset_path_train, 'training', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, args_swa.cycle_steps * args_swa.training_cycles, device, args.background_volume, args.background_frequency, args.silence_percentage, args.unknown_percentage, args.time_shift_ms)

speech_dataset_val = SpeechCommandsGoogle(args.dataset_path_train, 'validation', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0.)

speech_dataset_test = SpeechCommandsGoogle(args.dataset_path_train, 'testing', args.validation_percentage, args.testing_percentage, args.word_list, args.sample_rate, args.batch_size, epoch_list[-1], device, 0., 0., args.silence_percentage, args.unknown_percentage, 0., non_canonical_test = not args.canonical_testing)

train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
test_dataloader = torch.utils.data.DataLoader(speech_dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
validation_dataloader = torch.utils.data.DataLoader(speech_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)

model = KWS_LSTM(input_dim = args.n_mfcc, hidden_dim = args.hidden, output_dim = len(args.word_list), batch_size = args.batch_size, device = device, quant_factor = 1, quant_beta = 1, wb = args.quant_w, abMVM = args.quant_actMVM, abNM = args.quant_actNM, ib = args.quant_inp, noise_level = args.noise_injectionT, blocks = 1, pool_method = 'Max', fc_blocks = 0, cy_div = 1, cy_scale = 1).to(device)
model.load_state_dict(checkpoint_dict['model_dict'])
model.to(device)

model_uuid = str(uuid.uuid4())


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.00002, step_size_up = args_swa.cycle_steps, step_size_down = 1)

w_swa = model.state_dict()

n_models = 1

print("SWA Training")
print(model_uuid)
print("Collecting Models")
for e, (x_data, y_label) in enumerate(train_dataloader):
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda)
    output = model(x_data)
    
    loss_val = loss_fn(output, y_label)
    loss_val += args.l2 * torch.norm(torch.cat([model.lstmBlocks.cell.a1, model.lstmBlocks.cell.a3,  model.lstmBlocks.cell.a2,  model.lstmBlocks.cell.a4, model.lstmBlocks.cell.a5, model.lstmBlocks.cell.a6,  model.lstmBlocks.cell.a7,  model.lstmBlocks.cell.a8, model.lstmBlocks.cell.a9, model.lstmBlocks.cell.a10,  model.lstmBlocks.cell.a11, model.finFC.a1, model.finFC.a2]))
    train_acc = (output.argmax(dim=1) == y_label).float().mean().item()

    loss_val.backward()
    optimizer.step()
    optimizer.zero_grad()

    scheduler.step()

    if (e%args_swa.cycle_steps == 0) or (e == len(train_dataloader)-1):
        print("Cycle {0:03d}, Train Acc {1:.4f}".format(int(e/args_swa.cycle_steps), train_acc))
        new_w = model.state_dict()
        for w_name in w_swa:
            w_swa[w_name] = (w_swa[w_name] * n_models + new_w[w_name])/(n_models + 1)
        n_models += 1

# set model to average
model.load_state_dict(w_swa)

# evaluate model
acc_aux = []
for i_batch, sample_batch in enumerate(test_dataloader):
    x_data, y_label = sample_batch
    x_data, y_label = pre_processing(x_data, y_label, device, mfcc_cuda)

    output = model(x_data)
    acc_aux.append((output.argmax(dim=1) == y_label))

test_acc = torch.cat(acc_aux).float().mean().item()
print("Test Accuracy: {0:.4f}".format(test_acc))


checkpoint_dict = {
    'model_dict' : model.state_dict(), 
    'arguments'  : args_swa
}
torch.save(checkpoint_dict, './checkpoints/'+model_uuid+'.pkl')
