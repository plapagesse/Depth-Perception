from model import DepthPerception
import torch
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
from io import BytesIO
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
# Data stuff

def is_pil_image(img):
    return isinstance(img, Image.Image)

def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def visualize_pred(scene, depth, epoch):
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(scene)
    axs[1].imshow(depth, cmap='seismic')
    plt.savefig("results/" + epoch + ".png")

class InternalLoader(Dataset):
    def __init__(self, data, to_tensor=None):
        self.dataset = data
        self.transform = to_tensor

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        numpy_depth = sample['scene_depth']
        default_image = sample['scene']
        # image = Image.open(default_image)

        # array = np.load(numpy_depth)
        # depth = Image.fromarray(array)

        sample = {'image': default_image, 'depth': numpy_depth}
        if self.transform: sample = self.transform(sample) 
        
        return sample

    def __len__(self):
        return len(self.dataset)

def main():
    # Initialize model arch
    model = DepthPerception().cuda()
    summary(model, (3,768,1024))

    # Initialize optimizer
    batch = 5 
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    scan_ids_indoor = {"00019": ["00183"],
                    "00020": ["00184", "00185", "00186", "00187"],
                    "00021": ["00188", "00189", "00190", "00191", "00192"]}
    scan_ids_outdoor = {"00022": ["00193", "00194", "00195", "00196", "00197"],
                    "00023": ["00198", "00199", "00200"],
                    "00024": ["00201", "00202"]}
    scenes_scans = {"indoors" : scan_ids_indoor,
                "outdoor" : scan_ids_outdoor}
    dataset = []


    if os.path.exists('dataset.pkl'):
        with open('dataset.pkl', 'rb') as file:
            dataset = pickle.load(file)
    else: 
        for start in ["indoors", "outdoor"]:
            path = "data/" + start + '/'
            for scene in scenes_scans[start].keys():
                scene_path = path + "scene_" + scene + '/'
                for scan in scenes_scans[start][scene]:
                    scan_path = scene_path + "scan_" + scan + '/'
                    files = sorted(os.listdir(scan_path))
                    data = [files[i:i+3] for i in range(0, len(files), 3)]
                    for perspective in data:
                        scenen = torchvision.io.read_image(scan_path + perspective[0]).float()
                        scene_depth = torch.from_numpy(np.load(scan_path + perspective[1]))
                        scene_depth_mask = np.load(scan_path + perspective[2])
                        scene_depth_mask = torch.from_numpy(np.resize(scene_depth_mask,
                                                        (scene_depth_mask.shape[0], scene_depth_mask.shape[1], 1)))
                        scene_depth = (scene_depth*scene_depth_mask).movedim(-1,0)
                        scene_depth = torch.clamp(scene_depth, 0.4, 10)
                        dataset.append({"scene": scenen, "scene_depth": scene_depth})

        with open('dataset.pkl', 'wb') as file:
                pickle.dump(dataset, file)


    # load data
    tr_loader = DataLoader(InternalLoader(dataset), batch, shuffle = True)
    tst_loader = DataLoader(InternalLoader(dataset), batch, shuffle = True)

    l1_criterion = torch.nn.L1Loss()

    # set num epochs

    for epoch in tqdm(range(10)):

        model.train()

        for i, batched in enumerate(tr_loader):
            optimizer.zero_grad()
            #print(batched)

            default = batched['image'].cuda()
            depths = batched['depth'].cuda()

            # normalize depth ?? 

            prediction = model(default)


            l_depth = l1_criterion(prediction, depths)


            l_depth.backward()
            optimizer.step()

            print(l_depth)
            visualize_pred(default, prediction, epoch)





    pass

if __name__ == '__main__':
    main() 