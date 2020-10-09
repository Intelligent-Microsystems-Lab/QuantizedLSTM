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

import matplotlib.pyplot as plt

def plot_curves(train, val, f_name):
	plt.clf()
	plt.plot(train, label = "Train Acc")
	plt.plot(list(range(0, len(val)*100, 100)),val, label = "Val Acc")
	plt.xlabel("Epochs")
	plt.ylabel("Acc")
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/"+ f_name + ".png")
	plt.close()
