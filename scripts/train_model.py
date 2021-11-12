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
from datasetLoad import *
import importlib


# from model_res_enh import *

modellib = importlib.import_module('model_res_enh')
Model = modellib.Model
train = modellib.train
validate = modellib.validate
save_model = modellib.save_model

divider = "----------------------------------------------------------------\n"
min_size = (33, 42)
avg_size = (360, 360)
input_size = (180, 180)

transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

transform_up = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(avg_size),
     transforms.ToTensor()]
     )

transform_down = transforms.Compose(
    [transforms.CenterCrop(size=avg_size)]
    )

dataset_train = CatAndDogDataset('./data/train/', transform_upscale=transform_up, transform_downscale=transform_down) #, transform=transform, transform_upscale=transform_up, transform_downscale=transform_down
dataset_test = CatAndDogDataset('./data/test1/', transform_upscale=transform_up, transform_downscale=transform_down)

train_size = int(0.8 * len(dataset_train))
test_size = len(dataset_train) - train_size
dataset_train, dataset_validation = torch.utils.data.random_split(dataset_train, [train_size, test_size])

print(f"DataSet loaded!\n\nTrain set size: {len(dataset_train)}\nValidation size: {len(dataset_validation)}\nTest set size {len(dataset_test)}\n")

batch_size = 5

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}\n")

cur_model = Model()
cur_model = cur_model.to(device)

learning_rate = 1e-3
num_epochs = 5

optimizer = torch.optim.Adam(params=cur_model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.MSELoss()

print("Model definition: ")
summary(cur_model, (3, 180, 180), 1)

val_loss_avg = []
train_loss_avg = []
for t in range(num_epochs):
    print(f"Epoch {t+1}\n{divider}")
    train_loss = train(dataloader_train, cur_model, criterion, optimizer)
    val_loss = validate(dataloader_validation, cur_model, criterion)
    
    train_loss_avg.append(train_loss)
    val_loss_avg.append(val_loss)
    
    print(f"Train loss: {train_loss:>8f}\nTest Error: {val_loss:>8f} \n")
    
print(f"\n{divider}Done!")

test_loss = validate(dataloader_test, cur_model, criterion)
print('average reconstruction error: %f' % (test_loss))


current_model_path = save_model("REM", cur_model)