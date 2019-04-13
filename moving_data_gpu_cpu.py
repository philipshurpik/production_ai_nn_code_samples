import time

import numpy as np
import torch
from imageio import imread

iterations = 12
init_iterations = 2
image = imread("images/obama.jpg")
img_mean = torch.ByteTensor((118, 116, 117)).to('cuda:0')


def cpu_to_gpu(image):
    return torch.from_numpy(image).to('cuda:0')


def gpu_to_cpu(image_tensor):
    return image_tensor.cpu().numpy()


def ariphmetic(image_tensor):
    return image_tensor.div(255).sub(img_mean)


def ariphmetic_(image_tensor):
    return image_tensor.div_(255).sub_(img_mean)


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

measure(torch.tensor(imread("images/obama.jpg")).to("cuda:0").repeat(16,1,1,1), ariphmetic, "Ariphmetic: copy |")
measure(torch.tensor(imread("images/obama.jpg")).to("cuda:0").repeat(16,1,1,1), ariphmetic_, "Ariphmetic: inplace |")
