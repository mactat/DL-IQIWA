from random import randint
from dataset_load import CatAndDogDataset

from torchvision import transforms
from torch.utils.data import  DataLoader
import torch
import torch.nn as nn
from torchvision.utils import save_image
from random import randint
import argparse
import importlib


#parsing args
parser = argparse.ArgumentParser(description='Parameters for generating results')

parser.add_argument('--modeldef',  default="model",help='Specify the model file(without .py extension)')
parser.add_argument('--modelparams', help='Specify the model parameter file (with .pth extensoint')
parser.add_argument('--images',  type=int,  default=1, help='Specify the number of images')
parser.add_argument('--out', help='Specify output name (without .jpg)')

args = parser.parse_args()

try:
    modellib = importlib.import_module(args.modeldef)
    Model = modellib.Model
    train = modellib.train
    validate = modellib.validate
    save_model = modellib.save_model
except:
    raise Exception('Unable to import model lib!')

try:
    # Model.load_state_dict(torch.load(args.modelparams, map_location=torch.device('cpu')))
    model = Model()
    model.load_state_dict(torch.load(args.modelparams, map_location=torch.device('cpu')))
except:
    raise Exception(args.modelparams)

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

dataset_test = CatAndDogDataset('./data/test1/', transform_upscale=transform_up, transform_downscale=transform_down)
batch_size = 10

dataloader_validation = DataLoader(dataset_test, batch_size=batch_size)

print("Dataloader length: ", len(dataloader_validation))

for i, (image_batch) in enumerate(dataloader_validation):
    resized_img = transforms.Resize(input_size)(image_batch)
    pred = model(resized_img)

    before_img = image_batch[0]
    after_img = pred[0]
    concat_img = torch.cat((before_img, after_img), 2)
    save_image(concat_img, args.out + "_" + str(i) + ".jpeg")

    if i == args.images - 1:
        break


print(f"Saved {args.images} images !!!")
