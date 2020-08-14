import glob, os, argparse, re, hashlib, uuid, collections, math, time, imageio

from itertools import islice
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import skew

import matplotlib.pyplot as plt

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")


def basis_stack(s, T, past_obs_pred):
    s_ = s[:,[T-1] + list(range(0,T-1))] #basis functions (training)
    for i in range(past_obs_pred-1):
        s_ = torch.cat((s[:, [T-1] + list(range(0,T-1))], s_[:,[T-1] + list(range(0,T-1))]))
    return s_

def PCA(x, num_vec):
    z = x - x.mean(axis=0)
    ztz = z.t().mm(z)
    d, p = torch.eig(ztz, eigenvectors = True)

    return p[:, :num_vec].t()


training_samples = 1000 
Nimgx            = 144           # original image width
Nimgy            = 144           # original image height
Kp               = 40
Nu               = 80
prior_u          = 1


# create input sequence 
inputs = torch.zeros((Nimgy,Nimgy,3,training_samples*72))
test_inp = torch.zeros((Nimgy,Nimgy,3,1))
for n_ang in range(72):
    inputs[:,:,:,torch.arange(training_samples)*n_ang] = torch.stack(training_samples * [torch.tensor(imageio.imread("data.nosync/png4/62/62_r" + str((n_ang)*5) + ".png"), dtype = torch.uint8)[:,24:168,:] ]).permute(1,2,3,0).float()/255
    test_inp = inputs[:,:,:,n_ang]

s = inputs.sum(dim=2).view(144*144,-1)
s_test = test_inp.sum(dim=2).view(144*144,-1)

mean_s = inputs.mean(dim=0)
s = s - mean_s
s_test = s_test - mean_s


# Compute basis stack, Q, Wpca, A
s_    = basis_stack(inputs, 1000*72, Kp)
Q     = torch.mm(s.mm(s_.t()), torch.inverse(s_.mm(s_.t()) + torch.eye(Ns*Kp).to(device)*prior_s))
se    = Q.mm(s_)
Wppca = PCA(se.t(), Nu)
u     = Wpca.mm(se)
A     = torch.mm(s.t().mm(u.t()) , torch.inverse(u.mm(u.t()) + torch.eye(Nu).to(device)*prior_u))

# Predict on test
s__test = basis_stack(test_inp, 72, Kp)
se_test = Q.mm(s__test)
u_test  = Wpca.mm(se_test)
output  = A.mm(u_test.t()) + mean_s


#graph
plt.clf()
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(2,5)) 

for i in range(10):
    axes[i,0].imshow(inputs[i,:].view(144,144).cpu())
    axes[i,1].imshow(output[i,:].view(144,144).cpu())
    axes[i,0].axis('off')
    axes[i,1].axis('off')

plt.tight_layout()
plt.savefig("figures/duck.png")
plt.close()
