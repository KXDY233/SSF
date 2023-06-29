import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MLP(torch.nn.Module):
    # adopt a MLP as classifier for graphs
    def __init__(self,input_size, hidden_channels):
        super(MLP, self).__init__()
        self.nn = nn.BatchNorm1d(input_size)
        self.linear1 = torch.nn.Linear(input_size,hidden_channels*20)
        self.linear2 = torch.nn.Linear(hidden_channels*20,hidden_channels*10)
        self.linear3 = torch.nn.Linear(hidden_channels*10,hidden_channels)
        self.linear4 = torch.nn.Linear(hidden_channels,hidden_channels)
        self.linear5 = torch.nn.Linear(hidden_channels,1)
        self.act= nn.ReLU()

    def forward(self, x):
        out= self.nn(x)
        out= self.linear1(out)
        out = self.act(out)
        out= self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)
        out = F.dropout(out, p=0.7, training=self.training)
        out = self.linear5(out)
        return out
