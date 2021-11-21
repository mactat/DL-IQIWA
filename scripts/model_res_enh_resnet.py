from torch.utils.data import  DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
import glob
from random import randint
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchsummary import summary
import torch
import torch.nn as nn
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime


input_size = (180, 180)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)


class ResolutionEnhancerModel(nn.Module):
    def __init__(self):
        super(ResolutionEnhancerModel, self).__init__()     # input image size: 180 x 180
        self.up_sample = nn.Upsample(2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)    # out image size: 180 x180                                                       # out image size:  90 x 90
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)   # out image size:  359 x 359
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=4, stride=1, padding=1)   # out image size: 360 x 360

        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.eval()

        def ResnetCriterion(resnet, mse):
            def criterion(pred, original):
                return mse(resnet(pred), resnet(original))
            
            return criterion
        
        self.criterion = ResnetCriterion(self.resnet, nn.MSELoss())

            
    def forward(self, x):
        x = self.up_sample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x
    


def train(dataloader, model, loss_fn, optimizer, transform=transforms.Resize((180, 180))):
    size = len(dataloader.dataset)
    model.train()
    train_loss = []
    for batch, (original_image_batch) in enumerate(dataloader):
        if transform:
            t_image_batch = transform(original_image_batch)

        t_image_batch = t_image_batch.to(device)
        original_image_batch = original_image_batch.to(device)
        # Compute prediction error
        pred = model(t_image_batch)
        loss = loss_fn(pred, original_image_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(t_image_batch)
            train_loss.append(loss)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return (sum(train_loss))/len(train_loss)

def validate(dataloader, model, loss_fn, transform=transforms.Resize((180, 180))):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for original_image_batch in dataloader:
            if transform:
                t_image_batch = transform(original_image_batch)
            t_image_batch = t_image_batch.to(device)
            original_image_batch = original_image_batch.to(device)
            pred = model(t_image_batch)
            test_loss += loss_fn(pred, original_image_batch).item()
    test_loss /= num_batches
    
    return test_loss

def save_model(name, model, path='../model/'):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M")
    model_name = name + "_" + dt_string
    full_path = path+model_name+".pth"
    torch.save(model.state_dict(), full_path)
    
    print(f"Succesfully saved model!\nPath: {path}\nName: {model_name}\n")
    
    return full_path