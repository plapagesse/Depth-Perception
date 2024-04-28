import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
from torchvision.models import densenet169
import torchvision
from torchsummary import summary

# Outputs (num_channels,Height,Width) tensor
# densenet121 = densenet121(weights='DEFAULT')
# densenet121 = densenet121.cuda()
# densenet121.classifier = nn.Linear(densenet121.classifier.in_features, densenet121.classifier.in_features)
# scene1_depth = np.load("data/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010_depth.npy")
# summary(densenet121.features, (3,768,1024))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.features = densenet121(weights='DEFAULT').cuda().features
        self.relu = nn.ReLU()
    def forward(self,x):
        skips = []
        for layer in self.features:
            #print("x shape: ", x.shape)
            x = layer(x)
            skips.append(x)
        skips[11] = self.relu(skips[11])
        return [skips[i] for i in (2,3,5,7,11)] # Conv1, Pool1, Pool2, Pool3, x


# encoder = Encoder()
# skips = encoder.forward(scene1.float().resize(1,3,768,1024))
# print(skpis[4])
class UpSampleBlock(nn.Sequential):
    def __init__(self,concat_size,out_size):
        super(UpSampleBlock,self).__init__()
        self.convA = nn.Conv2d(concat_size,out_size,kernel_size=3,stride=1,padding=1)
        self.lr1 = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(out_size,out_size,kernel_size=3,stride=1,padding=1)
        self.lr2 = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')

    def forward(self,x,to_concat):
        x = torch.cat((self.upsample(x),to_concat),dim=1) #dim 1 is channels dimension, fisnal shape = (B, x_channel + skip_channels, H,W)
        x = self.lr2(self.convB(self.lr1(self.convA(x))))
        return x
class Decoder(nn.Module):
    def __init__(self,feature_channels = 1024):
        super(Decoder, self).__init__()

        self.up1 = UpSampleBlock(concat_size= feature_channels+256, out_size=feature_channels//2)
        self.up2 = UpSampleBlock(concat_size= feature_channels//2 + 128, out_size=feature_channels//4)
        self.up3 = UpSampleBlock(concat_size= feature_channels//4 + 64, out_size=feature_channels//8)
        self.up4 = UpSampleBlock(concat_size= feature_channels//8 + 64, out_size=feature_channels//16)
        self.conv2 = nn.Conv2d(feature_channels,feature_channels,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(feature_channels//16,1,kernel_size=3,stride=1,padding=1)
    def forward(self,skips):
        # Skips: Conv1, Pool1, Pool2, Pool3, x
        x = self.conv2(skips[4])
        x = self.up1(x,skips[3])
        x = self.up2(x,skips[2])
        x = self.up3(x,skips[1])
        x = self.up4(x,skips[0])
        x = self.conv3(x)
        return x

class DepthPerception(nn.Module):
    def __init__(self):
        super(DepthPerception, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self,x):
        return self.decoder(self.encoder(x))


model = DepthPerception()
model = model.cuda()
model.encoder.requires_grad_(False)
summary(model,(3,768,1024))
# scene1 = torchvision.io.read_image("data/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010.png")
# scene1 = scene1.cuda()
# d_map = model(scene1.float().resize(1,3,768,1024))
# print(d_map.shape)