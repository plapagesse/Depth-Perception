import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm

def normalize(tensor):
    return (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))


scene1 = torchvision.io.read_image("data/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010.png")
scene1_depth = np.load("data/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010_depth.npy")
scene1_depth_mask = np.load("data/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010_depth_mask.npy")
scene1_depth_mask = np.resize(scene1_depth_mask, (scene1_depth_mask.shape[0], scene1_depth_mask.shape[1], 1))
scene1_depth = scene1_depth * scene1_depth_mask
scene1 = scene1.movedim(0, -1)

scene1 = torch.cat((scene1,))

# scene1_depth = normalize(scene1_depth)
fig, axs = plt.subplots(ncols=2)
axs[0].imshow(scene1.float().int())
axs[1].imshow(scene1_depth, cmap='seismic')
plt.savefig("test.png")

# scan_ids_indoor = {"00019": ["00183"],
#                    "00020": ["00184", "00185", "00186", "00187"],
#                    "00021": ["00188", "00189", "00190", "00191", "00192"]}
# scan_ids_outdoor = {"00022": ["00193", "00194", "00195", "00196", "00197"],
#                     "00023": ["00198", "00199", "00200"],
#                     "00024": ["00201", "00202"]}
# scenes_scans = {"indoors" : scan_ids_indoor,
#                 "outdoor" : scan_ids_outdoor}
# full_data = []
# for start in tqdm(["indoors", "outdoor"]):
#     path = "data/" + start + '/'
#     for scene in scenes_scans[start].keys():
#         scene_path = path + "scene_" + scene + '/'
#         for scan in scenes_scans[start][scene]:
#             scan_path = scene_path + "scan_" + scan + '/'
#             files = os.listdir(scan_path)
#             data = [files[i:i+3] for i in range(0, len(files), 3)]
#             for perspective in data:
#                 scenen = torchvision.io.read_image(scan_path + perspective[0]).float()
#                 scene_depth = torch.from_numpy(np.load(scan_path + perspective[1]))
#                 scene_depth_mask = np.load(scan_path + perspective[2])
#                 scene_depth_mask = torch.from_numpy(np.resize(scene_depth_mask,
#                                              (scene_depth_mask.shape[0], scene_depth_mask.shape[1], 1)))
#                 scene_depth = (scene_depth*scene_depth_mask).movedim(-1,0)
#                 full_data.append({"scene": scenen, "scene_depth": scene_depth})
# print("done")
            # scene1 = torchvision.io.read_image(scan_path + data[0][0])
            # scene1 = scene1.movedim(0, -1)
            # scene_depth = np.load(scan_path + data[0][1])
            # scene_depth_mask = np.load(scan_path + data[0][2])
            # scene_depth_mask = np.resize(scene_depth_mask,
            #                               (scene_depth_mask.shape[0], scene_depth_mask.shape[1], 1))
            # scene_depth = scene_depth * scene_depth_mask
            # fig, axs = plt.subplots(ncols=2)
            # axs[0].imshow(scene1)
            # axs[1].imshow(normalize(scene_depth), cmap='seismic')
            # plt.savefig("results/" + scan + ".png")



