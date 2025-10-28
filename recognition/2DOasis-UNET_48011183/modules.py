import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

# Used for feature extraction
# Two Convolution sequences
class DoubleConv(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, 3, 1, 1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True),

            # Keeps channels constant
            nn.Conv2d(output_ch, output_ch, 3, 1, 1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True)
        )

    # Applies the convolution sequence to tensor
    def forward(self, x):
        return self.conv(x)
    

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            # Halves spatial size
            nn.MaxPool2d(2),
            # Extracts Features         
            DoubleConv(in_ch, out_ch)
        )

    # Applies pooling and Double Convolution sequence to input tensor
    def forward(self, x):
        return self.down(x)
    

# Upsamples the input from the previous decoder layer and refines features
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Learnable
            self.up = nn.ConvTranspose2d(in_ch // 2, out_ch // 2, 2, 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, y):
        # Upsamples decoder feature map
        x = self.up(x)

        # Pads x, y if sizes do not match
        map_x = y.size()[3] - x.size()[3]
        map_y = y.size()[2] - x.size()[2]
        
        x_pad = map_x // 2
        y_pad = map_y // 2

        x = f.pad(x, [x_pad, map_x - x_pad, y_pad, map_y - y_pad])

        # Refines features using DoubleConv
        return self.conv(torch.cat([y, x], 1))


# Reduces feature maps from in_ch to out_ch
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class ImprovedUNET(nn.Module):
    def __init__ (self, channels, classes, bilinear=True):
        super().__init__()
        self.channels = channels
        self.classes = classes
        self.bilinear = bilinear

        self.factor = 2 if self.bilinear else 1

        self.inconv = DoubleConv(channels, 64)
        self.outconv = OutConv(64, classes)

        self.encode()
        self.decode()

    def encode(self):
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        self.down4 = Down(512, 1024 // self.factor)
    
    def decode(self):
        self.up1 = Up(1024 // self.factor + 512, 512 // self.factor, self.bilinear)
        self.up2 = Up(512 // self.factor + 256, 256 // self.factor, self.bilinear)
        self.up3 = Up(256 // self.factor + 128, 128 // self.factor, self.bilinear)
        self.up4 = Up(128 // self.factor + 64, 64, self.bilinear)

    def forward (self, x):
        x1 = self.inconv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outconv(x)