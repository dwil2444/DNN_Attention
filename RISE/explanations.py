import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from utils.helper import GetDevice


def generate_masks(N, s, p1):
    cell_size = np.ceil(np.array((28 ,28)) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N,*(28, 28)))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + 28, y:y + 28]
    masks = masks.reshape(-1, *(28, 28), 1)
    return masks


def explain(model, inp, masks, N, p1):
    device = GetDevice()
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    masked = torch.tensor(masked)
    masked = masked.unsqueeze(1)
    for i in tqdm(range(0, N, 1), desc='Explaining'):
        masked = torch.tensor(masked).float()
        masked = masked.to(device)
        preds.append(model(masked[i:min(i+1, N)]).cpu().detach().numpy())
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *(28, 28))
    sal = sal / N / p1
    return sal