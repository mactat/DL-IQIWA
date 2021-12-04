import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
from datetime import datetime

input_size = (90, 90)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss,self).__init__()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()
        self.lambda_pixel = 100
    def __call__(self, pred, original):
        loss_GAN = self.criterion_GAN(pred, original)
        loss_pixel = self.criterion_pixelwise(pred, original)
        return loss_GAN + self.lambda_pixel * loss_pixel

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  

        # decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)   
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)                                             
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)   
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        # Loss functions

        self.criterion = MyLoss()

            
    def forward(self, x):
        #encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))

        #decoder
        x = F.relu(self.deconv1(x))
        x = self.upsample(x)
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

def train(dataloader, model, loss_fn, optimizer, transform=transforms.Resize(input_size)):
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


def validate(dataloader, model, loss_fn, transform=transforms.Resize(input_size)):
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
    
    print(f"Succesfully saved model!\nPath: {full_path}\nName: {model_name}\n")
    
    return model_name