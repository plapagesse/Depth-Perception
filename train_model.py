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

class ToTensor(object):
    def __init__(self,test=False):
        self.test = test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        depth = depth.resize((384, 512))
        
        # Can switch
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(is_pil_image(pic) or is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

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