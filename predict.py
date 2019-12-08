import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from arguments import *
from dataloader import AudioDataset
from model import RNN
import os



dataset = AudioDataset('data')
test_dataset = AudioDataset('test')
test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=False)


rnn = RNN(input_size, hidden_size, num_layers, num_classes)
rnn.load_state_dict(torch.load('./rnn.pth'))
iter_test = iter(test_loader)

with torch.no_grad():
        '''
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
        '''
        right = 0
        for i in range(len(test_dataset)):
            test_data, test_labels = iter_test.next()
            #print('so far so good')
            test_data = test_data.type(torch.float32)
            test_data, test_labels = test_data, test_labels

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