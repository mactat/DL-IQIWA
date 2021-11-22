from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
import glob
from torchvision import transforms
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CatAndDogDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_upscale=None, transform_downscale=None):
        self.imgs_path = root_dir
        self.file_list = glob.glob(str(self.imgs_path) + "*.jpg")
        self.data = []
        for img_path in self.file_list:
            self.data.append(img_path)

        self.transform = transform
        self.transform_up = transform_upscale
        self.transform_down = transform_downscale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        if self.transform_up:
            image = self.transform_up(image)
        if self.transform_down:
            image = self.transform_down(image)

        return image
