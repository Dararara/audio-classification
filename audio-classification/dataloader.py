import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import scipy.io.wavfile as wa
from collections import defaultdict
from python_speech_features import mfcc
import time
import numpy as np
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
        d_path = os.path.join(self.data_path, sub_class, self.audio_set[index][1])
        #print(type(waveform[0]))
        #temp = waveform
        #temp = np.random.randint(-5, 5, waveform.shape,dtype=int)
        #print(temp.shape)
        #temp = waveform + temp
        #print(waveform[:10])
        #print(temp[:10])
        #temp += (np.random.random(temp.shape) - 0.5) * 5
        temp = waveform    
        mfcc_feature = mfcc(temp, sample_rate, nfft= 1500, numcep=13)
        
        return mfcc_feature, self.name_to_index_dict[sub_class], d_path

    def get_label_by_index(self, index):
        index = int(np.array(index.cpu()))
        return self.index_to_name_dict[index]

    def __len__(self):
        return len(self.audio_set)







