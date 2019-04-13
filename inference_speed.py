import torch
from PIL import Image
import time
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = Image.open("images/obama.jpg")

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
    return preprocess(Image.open(image_path)).unsqueeze_(0).half().to(device)


def infer(image_batch):
    with torch.no_grad():
        return model.forward(image_batch).argmax()


batch_size = 16
result = infer(read_image("images/obama.jpg").repeat(batch_size, 1, 1, 1))
print(result)
time.sleep(10)
