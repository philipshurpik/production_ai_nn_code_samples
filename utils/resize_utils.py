import numpy as np
import pylru
import torch
import torch.nn.functional as F

DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
cache = pylru.lrucache(size=10)


def make_resize_grid(height, width, device=DEVICE, dtype=torch.float32):
    x = torch.linspace(-1, 1, height, device=device).view(-1, 1).repeat(1, width)
    y = torch.linspace(-1, 1, width, device=device).repeat(height, 1)
    if dtype == torch.float16:
        x = x.half()
        y = y.half()
    grid = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), 2)
    grid.unsqueeze_(0)
    return grid


def grid_resize(img_batch, size=None, grid=None, device=DEVICE, dtype=torch.float32):
    if np.issubdtype(type(size), np.floating):
        height, width = int(img_batch.shape[2] * size), int(img_batch.shape[3] * size)
    else:
        height, width = size[0], size[1]

    if grid is None and (height, width, device, dtype) in cache:
        grid = cache.peek((height, width, device, dtype))
    if grid is None:
        grid = make_resize_grid(height, width, device, dtype)
        cache[(height, width, device)] = grid

    return F.grid_sample(img_batch, grid.expand(img_batch.shape[0], -1, -1, -1))
