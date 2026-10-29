import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,stride=2),
            nn.ReLU()                 
        )
        self.conv = DoubleConv(out_channels*2, out_channels)
    def forward(self, x, features):
        out = self.up(x)
        out = torch.cat((out, features), dim=1)
        return self.conv(out)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=32):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)       # (B, 32, 128, 128)
        self.down1 = Down(base_channels, base_channels * 2)     # (B, 64, 64, 64)
        self.down2 = Down(base_channels * 2, base_channels * 4) # (B, 128, 32, 32)

        # Bottleneck
        self.bottleneck = Down(base_channels * 4, base_channels * 8) # (B, 256, 16, 16)

        # Decoder
        self.up1 = Up(base_channels * 8, base_channels * 4) # (B, 128, 32, 32)
        self.up2 = Up(base_channels * 4, base_channels * 2) # (B, 64, 64, 64)
        self.up3 = Up(base_channels * 2, base_channels)     # (B, 32, 128, 128)

        # Output
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Bottleneck
        x4 = self.bottleneck(x3)

        # Decoder + Skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.out(x)