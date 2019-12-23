from torch import nn
from torch.autograd import Variable
import torch
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) # 13, 60, 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),# 60, 2
            nn.ReLU(32),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )
        #self.fc = nn.Linear(hidden_size, num_classes)
        
    
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