#!./venv/bin/python
import torchvision
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn
import numpy as np
from nets.fmnet import Net
from utils.helper import GetDevice, CleanCuda
from torch import optim 


def train(model, device, optimizer, criterion ,dataloader, 
            num_epochs, weightdir):
    """
    params: model: the neural network to be trained

    params: device: object representing the device on 
                    which a torch tensor is allocated.

    params: optimizer: the optimization algorithm

    params: criterion: the loss function

    params: dataloader: the data loader for the input

    params: num_epochs: the number of epochs to train for
    """
    model.train()
    model = model.to(device)
    print('Training the model. Make sure that loss decreases after each epoch.\n')
    prev_loss = 0
    for e in range(num_epochs):
        cum_acc = 0
        total_train = 0
        running_loss =0
        for images, labels in dataloader:
            total_train += images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            log_ps = model(images);
            optimizer.zero_grad()
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            cum_acc += (torch.sum(labels==torch.argmax(log_ps, dim=1)))
            running_loss += loss.item()
        if (e == 0):
            prev_loss = running_loss
        if (running_loss <= prev_loss):
            torch.save(model.state_dict(), weightdir + 'params.pth')
            prev_loss = running_loss
        print(f"Accuracy: {cum_acc/total_train}")
        print(f"Training loss: {running_loss}")
            




def main():
    bs = 512
    lr = 0.001
    weightdir = './weights/'
    CleanCuda()
    device = GetDevice()
    transform=transforms.Compose([ transforms.ToTensor(), ])
    mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=bs, shuffle=True,)
    model = model.to(device)
    train(model, device, optimizer, criterion, trainloader, 10, weightdir)
    





if __name__ == '__main__':
    import argparse
    import os
    main()


