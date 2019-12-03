import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchaudio
from collections import defaultdict
from torch import nn
from python_speech_features import mfcc
from torch.autograd import Variable
import time


input_size = 13
hidden_size = 60
num_layers = 3
num_classes = 4
batch_size = 1
num_epochs = 2
learning_rate = 0.01

class AudioDataset(Dataset):
    def __init__(self, data_path):
        self.audio_set = []
        self.name_to_index_dict = defaultdict(int)
        self.index_to_name_dict = defaultdict(str)
        count = 0
        for i in os.listdir(data_path):
            self.name_to_index_dict[i] = count
            count += 1
            self.index_to_name_dict[count] = i
            for wav in os.listdir(os.path.join(data_path, i)):
                self.audio_set.append((i, wav))
        self.data_path = data_path
    def __getitem__(self, index):
        sub_class = self.audio_set[index][0]
        waveform, sample_rate = torchaudio.load(os.path.join(self.data_path, sub_class, self.audio_set[index][1]))
        #print(waveform.shape, self.audio_set[index][1])
        #print('load: ', waveform.shape)
        mfcc_feature = mfcc(waveform, sample_rate, nfft= 1256)
        #print(mfcc_feature.shape)
        
        
        return mfcc_feature, self.name_to_index_dict[sub_class]
    
    def __len__(self):
        return len(self.audio_set)




class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) # 13, 60, 2
        self.fc = nn.Linear(hidden_size, num_classes)# 60, 2
    
    def forward(self, x):
        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        
        
        #print(type(h0), type(c0), type(x))
        
        #print(h0.shape, c0.shape, x.shape) 
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        #print(out)
       # print (out.size())
        # Decode hidden state of last time step
        #print(out[:, -1, :].shape)
        out = self.fc(out[:, -1, :])  
        return out

loss_F=torch.nn.CrossEntropyLoss()
dataset = AudioDataset('data')
train_loader = DataLoader(dataset, batch_size= 1, shuffle= True)
test_dataset = AudioDataset('test')
test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=False)



rnn = RNN(input_size, hidden_size, num_layers, num_classes).cuda()
optimizer=torch.optim.Adam(rnn.parameters(), lr=0.01)


for epoch in range(100):
    iter_test = iter(test_loader)
    iter_data = iter(train_loader)
    for i in range(len(dataset)):
        audio, label = iter_data.next()
        #print(label)
        #print(audio.shape)
        audio = audio.reshape((audio.shape[1], 1, 13))
        audio = audio.type(torch.float32)
        audio, label = audio.cuda(), label.cuda()
        #print(audio.shape)
        
        output = rnn(audio)
        output = output[0].unsqueeze(0)
        #print(output, label)
        loss = loss_F(output, label)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

    with torch.no_grad():
        right = 0
        iter_data = iter(train_loader)
        for i in range(len(dataset)):
            test_data, test_labels = iter_data.next()
            #print('so far so good')
            test_data = test_data.type(torch.float32)
            test_data, test_labels = test_data.cuda(), test_labels.cuda()

            test_pred = rnn(test_data.view(-1, 1,13))
            #print('hello world')
            test_pred = test_pred[0]
            #print(test_pred.shape, test_label.shape)
            #print(test_pred)
            prob = torch.nn.functional.softmax(test_pred)
            #print(prob)
            pre_cls = torch.argmax(prob)
            if(pre_cls == test_labels[0]):
                right += 1
            if(i == 7): print(right)
        print('acc: ', right, '/', len(dataset))

        right = 0
        for i in range(len(test_dataset)):
            test_data, test_labels = iter_test.next()
            #print('so far so good')
            test_data = test_data.type(torch.float32)
            test_data, test_labels = test_data.cuda(), test_labels.cuda()

            test_pred = rnn(test_data.view(-1, 1,13))
            #print('hello world')
            test_pred = test_pred[0]
            #print(test_pred.shape, test_label.shape)
            #print(test_pred)
            prob = torch.nn.functional.softmax(test_pred)
            #print(prob)
            pre_cls = torch.argmax(prob)
            if(pre_cls == test_labels[0]):
                right += 1
            if(i == 7): print(right)
        print('acc: ', right, '/', len(test_dataset))
