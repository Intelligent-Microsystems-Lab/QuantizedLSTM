import glob, os, argparse, re, hashlib

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np




MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


#todo add silence + background data


class SpeechCommandsGoogle(Dataset):
    """Google Speech Command Dataset configured from Hello Edge"""

    def __init__(self, root_dir, train_test_val, val_perc, test_perc, words, sample_rate, transform=None):
        """
        Args:
            root_dir (string): Directory with all the recording.
            set (string): training/testing/validation
            val_perc: 
            test_perc:
            words:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.list_of_files = []
        self.list_of_labels = []
        self.list_of_y = []
        sub_dirs = [x[0].split('/')[-1] for x in os.walk(root_dir)][1:]
        for cur_dir in sub_dirs:
            if cur_dir[0] == "_":
                continue
            files_in_dir = glob.glob(root_dir + "/" + cur_dir + "/" + "*.wav")
            for cur_f in files_in_dir:
                if which_set(cur_f, val_perc, test_perc) == train_test_val:
                    self.list_of_files.append(cur_f)
                    if cur_dir not in words:
                        self.list_of_y.append(len(words))
                        self.list_of_labels.append('unknown')
                    else:
                        self.list_of_y.append(words.index(cur_dir))
                        self.list_of_labels.append(cur_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.train_test_val = train_test_val
        self.val_perc = val_perc
        self.test_perc = test_perc
        self.words = words
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.list_of_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        

        waveform, sample_rate = torchaudio.load(self.list_of_files[idx])
        if sample_rate != self.sample_rate:
            raise ValueError('Specified sample rate doesn\'t match sample rate in .wav file.')
        uniform_waveform = torch.zeros((1, self.sample_rate))
        uniform_waveform[0, :waveform.shape[1]] = uniform_waveform[0, :waveform.shape[1]] + waveform[0,:]

        if self.transform:
            waveform = self.transform(uniform_waveform)

        return waveform[0].t(), self.list_of_y[idx]

