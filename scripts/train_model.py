from torch.utils.data import  DataLoader
from torchvision import transforms
from torchinfo import summary

import torch
from dataset_load import *
import importlib
import argparse
import subprocess
import os

#parsing args
parser = argparse.ArgumentParser(description='Parameters for training')

parser.add_argument('--model',  default="model",help='Specify the model file(without .py extension)')
parser.add_argument('--epochs',  type=int,  default=5, help='Specify the number of epochs')
parser.add_argument('--verbose', type=bool, default=True, help='Print output or not')

args = parser.parse_args()

print(f"Model file ./{args.model}.py\nNumber of epochs:{args.epochs}\n")


try:
    modellib = importlib.import_module(args.model)
    Model = modellib.Model
    train = modellib.train
    validate = modellib.validate
    save_model = modellib.save_model
except:
    raise Exception('Unable to import model lib!') 

divider = "----------------------------------------------------------------\n"
min_size = (33, 42)
avg_size = (360, 360)
input_size = modellib.input_size

def log(inp):
    with open(f"../model/{args.model}.log", "a") as f:
        f.write(inp)

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

learning_rate = 1e-4
num_epochs = args.epochs

optimizer = torch.optim.Adam(params=cur_model.parameters(), lr=learning_rate, weight_decay=1e-5)

criterion = cur_model.criterion

print("Model definition: ")
model_stats = summary(cur_model, (batch_size, 3, input_size[0], input_size[1]), verbose=0)

log("Model definition: \n")
summary_str = str(model_stats)
print(summary_str+"\n")
log(summary_str+"\n")


val_loss_avg = []
train_loss_avg = []
try:
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n{divider}")
        log(f"Epoch {t+1}\n{divider}")
        train_loss = train(dataloader_train, cur_model, criterion, optimizer)
        val_loss = validate(dataloader_validation, cur_model, criterion)
        
        train_loss_avg.append(train_loss)
        val_loss_avg.append(val_loss)
        
        print(f"Train loss: {train_loss:>8f}\nTest Error: {val_loss:>8f} \n")
        log(f"Train loss: {train_loss:>8f}\nTest Error: {val_loss:>8f} \n")
except KeyboardInterrupt:
    print("Interupted, but model will be validated anyway!")   
    
print(f"\n{divider}Done!")

test_loss = validate(dataloader_test, cur_model, criterion)
print('average reconstruction error: %f' % (test_loss))
log('average reconstruction error: %f\n' % (test_loss))

model_file_name = save_model(args.model, cur_model)


abs_path = os.path.abspath("../model")
rc = subprocess.call(f"cd {abs_path} && ./push_to_arti.sh {args.model} {model_file_name} {args.model}", shell=True)

