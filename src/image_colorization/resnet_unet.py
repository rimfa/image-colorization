import torch 
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T
from .unet import DoubleConv, Up

class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        #Encoder
        resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT) 
        for param in resnet_model.parameters():
            param.requires_grad = False
        
        self.conv1 = resnet_model.conv1 #[B, 3, 128, 128] -> [B, 64, 64, 64]
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool  

        self.layer1 = resnet_model.layer1 # [B, 64, 32, 32] -> [B, 64, 32, 32]
        self.layer2 = resnet_model.layer2 # [B, 64, 32, 32] -> [B, 128, 16, 16]
        self.layer3 = resnet_model.layer3 # [B, 128, 16, 16] -> [B, 256, 8, 8]
        self.layer4 = resnet_model.layer4 # [B, 256, 8, 8] -> [B, 512, 4, 4]

        #Decoder
        self.up1 = Up(in_channels=512, out_channels=256) # [B, 512, 4, 4] -> [B, 256, 8, 8]
        self.up2 = Up(in_channels=256, out_channels=128) # [B, 256, 8, 8] -> [B, 128, 16, 16]
        self.up3 = Up(in_channels=128, out_channels=64) # [B, 128, 16, 16] -> [B, 64, 32, 32]
        self.up4 = Up(in_channels=64, out_channels=64) # [B, 64, 32, 32] -> [B, 64, 64, 64]

        #Output
        # [B, 64, 64, 64] -> [B, 2, 128, 128]
        self.tconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.doubleconv1 = DoubleConv(in_channels=32, out_channels=32)
        self.outconv = nn.Conv2d(32, 2, kernel_size=1)
    def train(self, mode=True):
        super().train(mode)

        self.bn1.eval()
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            layer.eval()
        return self
    
    def forward(self, x):

        x = x.repeat(1, 3, 1, 1)
        x = self.normalize(x)

        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3) 

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)

        x9 = self.up1(x8, x7)
        x10= self.up2(x9, x6)
        x11= self.up3(x10, x5)
        x12= self.up4(x11, x3)

        x13 = self.tconv1(x12)
        x14 = self.doubleconv1(x13)
        x15 = self.outconv(x14)

        return x15



        