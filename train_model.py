from model import DepthPerception
import torch
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
import torchvision
from PIL import Image
from io import BytesIO
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import torchmetrics



# Data stuff

def is_pil_image(img):
    return isinstance(img, Image.Image)


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def visualize_pred(scene, pred_depth, real_depth, epoch):
    fig, axs = plt.subplots(ncols=3)
    scene = scene.cpu()
    pred_depth = pred_depth.cpu().detach()
    real_depth = real_depth.cpu()
    scene = scene.movedim(0, -1)
    pred_depth = pred_depth.movedim(0, -1)
    real_depth = real_depth.movedim(0, -1)
    axs[0].imshow(scene.int())
    pred_depth = torch.clamp(pred_depth, min=0.4, max=10)
    hm1 = axs[1].imshow(pred_depth, cmap='seismic')
    fig.colorbar(hm1)
    hm2 = axs[2].imshow(real_depth, cmap='seismic')
    fig.colorbar(hm2)
    plt.savefig("results/" + str(epoch) + ".png")
    plt.close(fig)


class InternalLoader(Dataset):
    def __init__(self, data, to_tensor=None):
        self.dataset = data
        self.transform = to_tensor

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        numpy_depth_mask = sample['scene_depth_mask']
        numpy_depth = sample['scene_depth']
        default_image = sample['scene']
        # image = Image.open(default_image)

        # array = np.load(numpy_depth)
        # depth = Image.fromarray(array)

        sample = {'image': default_image, 'depth': numpy_depth, 'depthmask': numpy_depth_mask}
        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.dataset)


def main():
    # Initialize model arch
    model = DepthPerception().cuda()
    summary(model, (3, 768, 1024))

    # Initialize optimizer
    batch = 2
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    scan_ids_indoor = {"00019": ["00183"],
                       "00020": ["00184", "00185", "00186", "00187"],
                       "00021": ["00188", "00189", "00190", "00191", "00192"]}
    scan_ids_outdoor = {"00022": ["00193", "00194", "00195", "00196", "00197"],
                        "00023": ["00198", "00199", "00200"],
                        "00024": ["00201", "00202"]}
    scenes_scans = {"indoors": scan_ids_indoor,
                    "outdoor": scan_ids_outdoor}
    dataset = []

    if os.path.exists('dataset.pkl'):
        with open('dataset.pkl', 'rb') as file:
            print("reading from pickle")
            dataset = pickle.load(file)
    else:
        # for start in ["indoors", "outdoor"]:
        for start in ["indoors"]:
            path = "data/" + start + '/'
            for scene in scenes_scans[start].keys():
                scene_path = path + "scene_" + scene + '/'
                for scan in scenes_scans[start][scene]:
                    scan_path = scene_path + "scan_" + scan + '/'
                    files = sorted(os.listdir(scan_path))
                    data = [files[i:i + 3] for i in range(0, len(files), 3)]
                    for perspective in data:
                        scenen = torchvision.io.read_image(scan_path + perspective[0]).float()
                        scene_depth = torch.from_numpy(np.load(scan_path + perspective[1]))
                        scene_depth_mask = np.load(scan_path + perspective[2])
                        scene_depth_mask = torch.from_numpy(np.resize(scene_depth_mask,
                                                                      (scene_depth_mask.shape[0],
                                                                       scene_depth_mask.shape[1], 1)))
                        scene_depth = (scene_depth * scene_depth_mask).movedim(-1, 0)
                        if start == "indoors":
                            scene_depth = torch.clamp(scene_depth, 0.4, 10)
                        else:
                            scene_depth = torch.clamp(scene_depth, 0.4, 80)
                        dataset.append({"scene": scenen, "scene_depth": scene_depth,
                                        "scene_depth_mask": scene_depth_mask.movedim(-1, 0)})

        with open('dataset.pkl', 'wb') as file:
            pickle.dump(dataset, file)

    # load data
    print("training loader")
    tr_loader = DataLoader(InternalLoader(dataset), batch, shuffle=True)
    print("test loader")
    tst_loader = DataLoader(InternalLoader(dataset), batch, shuffle=True)
    print("l1loss")
    l1_criterion = torch.nn.L1Loss()

    # set num epochs
    print("starting training")
    model.train()
    loader = list(tr_loader.batch_sampler)
    # Set up sobel/gradient convolutions
    dx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dx_nn = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    dx_nn.weight = nn.Parameter(dx.float().unsqueeze(0).unsqueeze(0))
    dx_nn.eval()
    dx_nn = dx_nn.cuda()

    dy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dy_nn = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    dy_nn.weight = nn.Parameter(dy.float().unsqueeze(0).unsqueeze(0))
    dy_nn.eval()
    dy_nn = dy_nn.cuda()

    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure().cuda()


    for epoch in tqdm(range((len(loader) - 1) * 2)):  # Go through dataset twice
        optimizer.zero_grad()
        batch = loader[epoch % len(loader)]
        default = []
        depths = []
        masks = []
        for i in batch:
            default.append(tr_loader.dataset[i]['image'].cuda())
            depths.append(tr_loader.dataset[i]['depth'].cuda())
            masks.append(tr_loader.dataset[i]['depthmask'].cuda())
        default = torch.stack(default)
        depths = torch.stack(depths)
        masks = torch.stack(masks)
        prediction = model(default)
        prediction = F.upsample(prediction, scale_factor=2, mode='bilinear')
        for i in range(len(batch)):
            prediction[i][masks[i] == 0] = 0.4
        l_depth = l1_criterion(prediction, depths)

        G_X = dx_nn(depths.float()).detach()
        G_Y = dy_nn(depths.float()).detach()
        G_X_p = dx_nn(prediction.float()).detach()
        G_Y_p = dy_nn(prediction.float()).detach()
        l_grad_x = l1_criterion(G_X_p,G_X)
        l_grad_y = l1_criterion(G_Y_p, G_Y)
        l_grad = l_grad_x + l_grad_y

        l_ssim =  (1 - ssim(prediction,depths))/2

        loss = l_ssim + l_grad + (0.1 * l_depth)


        loss.backward()
        optimizer.step()

        if (epoch % 10) == 0:
            print("SSIM_loss: ",l_ssim)
            print("Grad_loss ", l_grad)
            print("Depth loss/10 ", l_depth*0.1)
            visualize_pred(default[0], prediction[0], depths[0], epoch)
        # for batched in list(tr_loader.batch_sampler)[epoch]:
        #     optimizer.zero_grad()
        #     #print(batched)
        #
        #     default = tr_loader.dataset[]['image'].cuda()
        #     depths = tr_loader.dataset[batched]['depth'].cuda()
        #
        #     # normalize depth ??
        #
        #     prediction = model(default)
        #
        #     prediction = F.upsample(prediction,scale_factor=2,mode='bilinear')
        #     l_depth = l1_criterion(prediction, depths)
        #
        #
        #
        #     l_depth.backward()
        #     optimizer.step()
        #
        #     print(l_depth)
        #     visualize_pred(default[0], prediction[0], epoch)

    pass


if __name__ == '__main__':
    main()
