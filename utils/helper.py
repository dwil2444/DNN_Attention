import torchvision
import torchvision.transforms.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn


def GetDevice():
    if torch.cuda.is_available():       
        device = torch.device("cuda:0")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def CleanCuda():
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()