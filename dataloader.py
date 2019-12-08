import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import scipy.io.wavfile as wa
from collections import defaultdict
from python_speech_features import mfcc
import time




class AudioDataset(Dataset):
    def __init__(self, data_path):
        self.audio_set = []
        self.name_to_index_dict = defaultdict(int)
        self.index_to_name_dict = defaultdict(str)
        count = 0
        for i in os.listdir(data_path):
            
            self.name_to_index_dict[i] = count
            self.index_to_name_dict[count] = i
            for wav in os.listdir(os.path.join(data_path, i)):
                self.audio_set.append((i, wav))
            count += 1
        self.data_path = data_path
    def __getitem__(self, index):
        sub_class = self.audio_set[index][0]
        sample_rate,waveform = wa.read(os.path.join(self.data_path, sub_class, self.audio_set[index][1]))
        mfcc_feature = mfcc(waveform, sample_rate, nfft= 1256)
        
        
        
        return mfcc_feature, self.name_to_index_dict[sub_class]
    
    def __len__(self):
        return len(self.audio_set)







