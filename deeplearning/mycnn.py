import torch
import torch.nn as nn
import torch.optim as optim

class MyCnn(nn.model):
    def __init__(self):
        super(MyCnn,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,68*8*8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = MyCnn()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epoch = 10
for i in range(epoch):
    model.train()
