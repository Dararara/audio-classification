import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from arguments import *
from dataloader import AudioDataset
from model import RNN
import os

loss_F=torch.nn.CrossEntropyLoss()
dataset = AudioDataset('data')
train_loader = DataLoader(dataset, batch_size= 1, shuffle= True)




rnn = RNN(input_size, hidden_size, num_layers, num_classes)
optimizer=torch.optim.Adam(rnn.parameters(), lr=0.01)


for epoch in range(5):
    iter_data = iter(train_loader)
    for i in range(len(dataset)):
        audio, label = iter_data.next()
        #prcint(label)
        #print(audio.shape)
        audio = audio.reshape((audio.shape[1], 1, 13))
        audio = audio.type(torch.float32)
        audio, label = audio, label
        #print(audio.shape)
        
        output = rnn(audio)
        output = output[0].unsqueeze(0)
        #print(output, label)
        loss = loss_F(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(rnn.state_dict(), './rnn.pth')
    