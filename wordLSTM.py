import glob, os, argparse, re, hashlib

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-path", type=str, default='data/speech_commands_v0.01', help='path to the dataset')
args = parser.parse_args()

#MFCC feature extraction
#40 MFCC speech frame of length 40ms with a stride of 20ms, which gives 1960 (49×40) features for 1 second of audio


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

    def __init__(self, root_dir, train_test_val, val_perc, test_perc, words, transform=None):
        """
        Args:
            root_dir (string): Directory with all the recording.
            set (string): train/test/validation
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.list_of_files = []
        self.list_of_labels = []
        sub_dirs = [x[0].split('/')[-1] for x in os.walk(root_dir)][1:]
        for cur_dir in sub_dirs:
            if cur_dir[0] == "_":
                continue
            files_in_dir = glob.glob(root_dir + "/" + cur_dir + "/" + "*.wav")
            for cur_f in files_in_dir:
                if which_set(cur_f, val_perc, test_perc) == train_test_val:
                    self.list_of_files.append(cur_f)
                    if cur_dir not in words:
                        self.list_of_labels.append('unknown')
                    else:
                        self.list_of_labels.append(cur_dir)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_of_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        import pdb; pdb.set_trace()

        #for i in idx:
        waveform, _ = torchaudio.load(self.list_of_files[idx])


        
        

        #img_name = os.path.join(self.root_dir,
        #                        self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, self.list_of_labels[idx]




files_path = "data.nosync/speech_commands_v0.01"
word_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence'] #unknown category


data_transform = transforms.Compose([
        torchaudio.transforms.MFCC(sample_rate = 16000, n_mfcc = 40, melkwargs = {'win_length' : 40, 'hop_length' : 20})
    ])

speech_dataset_train = SpeechCommandsGoogle(files_path, 'training', 10, 10, word_list)
speech_dataset_test = SpeechCommandsGoogle(files_path, 'testing', 10, 10, word_list)
speech_dataset_val = SpeechCommandsGoogle(files_path, 'validation', 10, 10, word_list)
print(len(speech_dataset_train))
print(len(speech_dataset_test))
print(len(speech_dataset_val))

train_dataloader = torch.utils.data.DataLoader(speech_dataset_train, batch_size=4, shuffle=True, num_workers=4)


for i_batch, sample_batched in enumerate(train_dataloader):
    print(sample_batched)



