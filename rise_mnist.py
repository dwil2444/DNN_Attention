#!./venv/bin/python
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import numpy as np
from utils.helper import GetDevice, CleanCuda
from nets.fmnet import Net


device = GetDevice();
print(device)

fmnet = Net()

print(fmnet)