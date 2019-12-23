import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from arguments import *
from dataloader import AudioDataset
from model import RNN
import os
from predict import predict
loss_F=torch.nn.CrossEntropyLoss()
dataset = AudioDataset('data')
train_loader = DataLoader(dataset, batch_size= 1, shuffle= True)


rnn = RNN(input_size, hidden_size, num_layers, num_classes).cuda()
optimizer=torch.optim.Adam(rnn.parameters(), lr=0.01)


for epoch in range(30):
    iter_data = iter(train_loader)
    print(epoch)
    for i in range(len(dataset)):
        audio, label, d_path = iter_data.next()
        #prcint(label)
        #print(audio.shape)

        audio = audio.reshape((audio.shape[1], 1, 13))
        audio = audio.type(torch.float32)
        audio, label = audio.cuda(), label.cuda()
        #print(audio.shape)
        
        output = rnn(audio).cuda()
        output = output[0].unsqueeze(0)
        #print(output, label)
        loss = loss_F(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch % 10 == 0):
        predict(rnn, 'test')
        predict(rnn, 'data')

torch.save(rnn.state_dict(), './rnn.pth')