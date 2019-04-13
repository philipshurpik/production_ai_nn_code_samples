import time

import numpy as np
import torch
from imageio import imread

iterations = 12
init_iterations = 2
image = imread("images/obama.jpg")


def cpu_to_gpu(image):
    return torch.from_numpy(image).to('cuda:0')


def gpu_to_cpu(image_tensor):
    return image_tensor.cpu().numpy()


def measure(image, method, name):
    times = []
    for i in range(iterations):
        time1 = time.time()
        method(image)
        time2 = time.time()
        times.append([time2 - time1])
    print(f"{name} Total speed: {(np.sum(times[init_iterations:]) / (iterations - init_iterations))} sec")


measure(imread("images/obama.jpg"), cpu_to_gpu, "CPU to GPU: image |")
measure(torch.tensor(imread("images/obama.jpg")).to("cuda:0"), gpu_to_cpu, "GPU to CPU: image |")
