from torch.nn.functional import interpolate
from torchvision.transforms.functional import InterpolationMode
from dataset_load import *
from torch.utils.data import  DataLoader
import importlib
import torch.nn as nn
from kornia.losses.ssim import SSIMLoss
from kornia.losses.psnr import PSNRLoss, psnr_loss

from resnet import Resnet
from tnn import Tnn
input_size = (90, 90)
avg_size = (360, 360)


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

dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

bestModel = Resnet()
transposeModel = Tnn()

bestModel.load_state_dict(torch.load("resnet.pth"))
transposeModel.load_state_dict(torch.load("tnn.pth"))
bestModel.to(device)
transposeModel.to(device)


bestModel.eval()
transposeModel.eval()

mseLoss = nn.MSELoss()
psnrLoss = PSNRLoss(1)
ssimLoss = SSIMLoss(5)

MSE_losses  = [0, 0, 0, 0]
SSIM_losses = [0, 0, 0, 0]
PSNR_losses = [0, 0, 0, 0]

resize_t = transforms.Resize(input_size)
resize_t_180 = transforms.Resize((90, 90))
resize_t_nearest = transforms.Resize((360, 360), interpolation=InterpolationMode.NEAREST)
resize_t_bicubic = transforms.Resize((360, 360), interpolation=InterpolationMode.BICUBIC)

print("Generating losses")
for i, image_batch in enumerate(dataloader_test):
    image_batch = image_batch.to(device)
    scaled_down_180 = resize_t_180(image_batch)
    scaled_down_90 = resize_t(image_batch)

    scaled_up_nearest = resize_t_nearest(scaled_down_90)
    scaled_up_bicubic = resize_t_bicubic(scaled_down_90)
    scaled_up_bestModel = bestModel(scaled_down_90)
    scaled_up_transposeModel = transposeModel(scaled_down_180)

    MSE_losses[0] = (MSE_losses[0]*i +  mseLoss(scaled_up_nearest, image_batch).item())/(i+1)
    MSE_losses[1] = (MSE_losses[1]*i +  mseLoss(scaled_up_bicubic, image_batch).item())/(i+1)
    MSE_losses[2] = (MSE_losses[2]*i +  mseLoss(scaled_up_transposeModel, image_batch).item())/(i+1)
    MSE_losses[3] = (MSE_losses[3]*i +  mseLoss(scaled_up_bestModel, image_batch).item())/(i+1)

    SSIM_losses[0] = (SSIM_losses[0]*i +  ssimLoss(scaled_up_nearest, image_batch).item())/(i+1)
    SSIM_losses[1] = (SSIM_losses[1]*i +  ssimLoss(scaled_up_bicubic, image_batch).item())/(i+1)
    SSIM_losses[2] = (SSIM_losses[2]*i +  ssimLoss(scaled_up_transposeModel, image_batch).item())/(i+1)
    SSIM_losses[3] = (SSIM_losses[3]*i +  ssimLoss(scaled_up_bestModel, image_batch).item())/(i+1)

    PSNR_losses[0] = (PSNR_losses[0]*i +  psnrLoss(scaled_up_nearest, image_batch).item())/(i+1)
    PSNR_losses[1] = (PSNR_losses[1]*i +  psnrLoss(scaled_up_bicubic, image_batch).item())/(i+1)
    PSNR_losses[2] = (PSNR_losses[2]*i +  psnrLoss(scaled_up_transposeModel, image_batch).item())/(i+1)
    PSNR_losses[3] = (PSNR_losses[3]*i +  psnrLoss(scaled_up_bestModel, image_batch).item())/(i+1)

    if i % 1 == 0:
        print("MSE losses")
        print(MSE_losses)
        print("PSNR losses")
        print(PSNR_losses)
        print("SSIM losses")
        print(SSIM_losses)



for j in range(len(MSE_losses)):
    MSE_losses[i] = MSE_losses[i]/len(dataloader_test) 
    PSNR_losses[i] = MSE_losses[i]/len(dataloader_test) 
    SSIM_losses[i] = MSE_losses[i]/len(dataloader_test)


print("MSE losses")
print(MSE_losses)
print("PSNR losses")
print(PSNR_losses)
print("SSIM losses")
print(SSIM_losses)
