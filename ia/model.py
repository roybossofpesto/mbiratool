
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, nclass_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 128)
        #self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(129, 64)
        self.fc2 = nn.Linear(64, nclass_out)
        self.fc3 = nn.Linear(nclass_out, nclass_out)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print('forward', x.size())
        x = x.view(-1, 129)
        #x = self.pool(x, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

Criterion = nn.CrossEntropyLoss
#Criterion = nn.MSELoss

Optimizer = optim.Adadelta
#Optimizer = optim.SGD
