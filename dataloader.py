import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchaudio
from collections import defaultdict


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
       
        waveform = waveform.reshape([-1])
        
        if len(waveform) > 192000:
            waveform = waveform[0:192000]
        padding = torch.zeros(192000 - len(waveform))
        waveform = torch.cat((waveform, padding), 0)
        # print(torch.max(waveform), torch.min(waveform))
        # print(waveform.shape)
        return waveform, self.name_to_index_dict[sub_class]
    
    def __len__(self):
        return len(self.audio_set)


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size = 256,
            hidden_size=64,
            num_layers=32,
            batch_first=True
        )
        self.linear = torch.nn.Linear(in_features = 192000, out_features = 256)
        self.out=torch.nn.Linear(in_features=64,out_features=2)

    def forward(self,x):
        #print('input shape: ',x.shape)
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        x = self.linear(x)
        output,(h_n,c_n)=self.rnn(x)
        #print(output.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep=h_n[-1,:,:]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x=self.out(output_in_last_timestep)
        #print('output shape:', x.shape)
        return x


loss_F=torch.nn.CrossEntropyLoss()
dataset = AudioDataset('data')
train_loader = DataLoader(dataset, batch_size= 1, shuffle= True, num_workers= 2)
test_dataset = AudioDataset('test')
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
iter_test = iter(test_loader)
test_data, test_labels = iter_test.next()

rnn = RNN()
optimizer=torch.optim.Adam(rnn.parameters(),lr=0.001)


for epoch in range(5):
    iter_data = iter(train_loader)
    for i in range(len(dataset)):
        audio, label = iter_data.next()
        output = rnn(audio.view(-1, 1, 192000))
        loss = loss_F(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_pred = rnn(test_data.view(-1, 1, 192000))
        print('hello world')
        print(test_pred.shape, test_labels.shape)
        prob = torch.nn.functional.softmax(test_pred, dim = 1)
        print(prob)
        pre_cls = torch.argmax(prob, dim = 1)
        acc = (pre_cls == test_labels).sum().numpy()/pre_cls.size()[0]
        print('accuracy is: ', acc)