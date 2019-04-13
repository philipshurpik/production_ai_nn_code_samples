import time

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = Image.open("images/obama.jpg")

init_iterations = 2
iterations = 100

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
model = models.resnet18(pretrained=True).half().eval().to(device)


def read_image(image_path):
    return preprocess(Image.open(image_path)).half().unsqueeze_(0).to(device)


def infer(image_batch):
    with torch.no_grad():
        return model.forward(image_batch).argmax()


def measure(image, method, name):
    times = []
    results = []
    for i in range(iterations):
        time1 = time.time()
        result = method(image)
        time2 = time.time()
        times.append([time2 - time1])
        results.append(result)
    print(np.sum(times[init_iterations:]))
    print(f"{name} Total speed: {1 / (np.sum(times[init_iterations:]) / (iterations - init_iterations))} FPS")


batch_size = 16
image_batch = read_image("images/obama.jpg").repeat(batch_size, 1, 1, 1)
measure(image_batch, infer, "Half |")
