from model import DepthPerception
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import numpy as np
import os

# Data stuff

def is_pil_image(img):
    return isinstance(img, Image.Image)

def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def get_transform():
    return transforms.Compose([
        ToTensor()
    ])

class InternalLoader(Dataset):
    def __init__(self, data, to_tensor=None):
        self.dataset = data
        self.transform = to_tensor

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        numpy_depth = sample[1]
        default_image = sample[0]
        image = Image.open(default_image)

        array = np.load(numpy_depth)
        depth = Image.fromarray(array)

        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample) 
        
        return sample

    def __len__(self):
        return len(self.dataset)

def main():
    # Initialize model arch
    model = DepthPerception().cuda()

    # Initialize optimizer
    batch = 5 
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    dataset = []

    for dir in os.listdir('data'):
        dir_path = os.path.join('data', dir)

        for scene in os.listdir(dir_path):
            scene_path = os.path.join(dir_path, scene)

            for scan in os.listdir(scene_path):
                scan_path = os.path.join(scene_path, scan)


                for file in os.listdir(scan_path):
                    print(file)


    # load data
    # tr_loader = 
    # tst_loader =

    # log ? 

    # loss

    # train

    pass

if __name__ == '__main__':
    main() 