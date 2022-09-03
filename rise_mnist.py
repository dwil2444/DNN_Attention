#!./venv/bin/python
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import numpy as np
import torchvision.datasets as datasets
from utils.helper import GetDevice, CleanCuda
from nets.fmnet import Net
from torch import optim 
import torch
from RISE.explanations import generate_masks, explain
import torchvision.transforms as transforms


def explain_instance():
    """
    """
    device = GetDevice()
    model = Net().to(device)
    model.load_state_dict(torch.load('./weights/params.pth'))
    model.eval()
    transform=transforms.Compose([
        transforms.ToTensor(),])
    mnist_valset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    N = 200000
    s = 8
    p1 = 0.5
    masks = generate_masks(N, s, p1)
    masks = masks.squeeze(3)
    img = mnist_valset[1][0]
    label = mnist_valset[1][1]
    sal = explain(model, img, masks, N, p1)
    ans = sal[label]
    disp = img.squeeze().cpu().detach().numpy()
    plt.imshow(disp, cmap='gray')
    plt.imshow(ans, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('view_softmax.png')
    # retun


def main():
    explain_instance()


if __name__ == "__main__":
    main()
