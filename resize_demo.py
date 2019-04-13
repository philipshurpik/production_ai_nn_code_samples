import time

import torch
import numpy as np
from imageio import imread
from scipy.misc import imresize

from utils.resize_utils import grid_resize

iterations = 12
init_iterations = 2
big_image = imread("images/trump.jpg")
med_image = imread("images/obama.jpg")


def cpu_resize(image):
    return imresize(image, 0.4)


def gpu_resize(image):
    return grid_resize(image, 0.4)


def measure(image, method, name):
    times = []
    for i in range(iterations):
        time1 = time.time()
        method(image)
        time2 = time.time()
        times.append([time2 - time1])
    print(f"{name} Total speed: {1 / (np.sum(times[init_iterations:]) / (iterations - init_iterations))} FPS")


torch_big_image = torch.from_numpy(big_image).unsqueeze(0).permute(0, 3, 1, 2).float().to('cuda:0')
torch_med_image = torch.from_numpy(big_image).unsqueeze(0).permute(0, 3, 1, 2).float().to('cuda:0')

measure(big_image, cpu_resize, "CPU: big image |")
measure(torch_big_image, gpu_resize, "GPU: big image |")

measure(med_image, cpu_resize, "CPU: medium image |")
measure(torch_med_image, gpu_resize, "GPU: medium image |")
