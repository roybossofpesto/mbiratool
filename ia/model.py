
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, nsample, nclass_out, nmiddle, nkernel):
        super(Net, self).__init__()
        self.params = params
        self.kernel_size = nkernel
        self.linear_size = 1 + nsample - nkernel
        #print(self.kernel_size, self.linear_size)
        self.conv1 = nn.Conv1d(1, 1, self.kernel_size)
        #self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(self.linear_size, nmiddle)
        self.fc2 = nn.Linear(nmiddle, nclass_out)
        self.fc3 = nn.Linear(nclass_out, nclass_out)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print('forward', x.size())
        x = x.view(-1, self.linear_size)
        #x = self.pool(x, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

Criterion = nn.CrossEntropyLoss
#Criterion = nn.MSELoss

Optimizer = optim.Adadelta
#Optimizer = optim.SGD
