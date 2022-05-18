import torchvision
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(30*13*13, 400)  # input -> first hidden layer
        
        self.hidden_next = nn.Linear(400, 120) # first hidden -> second hidden layer
        # Output layer, 10 units - one for each digit

        
        self.output = nn.Linear(120, 10)  # second hidden -> output layer

        self.conv = nn.Conv2d(1, 30, 5, stride=1, padding=1)

        self.MaxPool2d = nn.MaxPool2d(2, 2)
         
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # apply softmax to logits

        
    def forward(self, x):
        print(x.shape)
        x = self.relu(self.conv(x))
        x = self.MaxPool2d(x)
        x = x.view(-1, 30* 13*13)
        x = self.hidden(x)
        x = self.hidden_next(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)     
        return x