import torch
import torch.nn as nn
import torch.nn.functional as f

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
            self.up = nn.Sequential(
                # Non-learnable scaling
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, 1)
            )
        else:
            # Learnable
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        
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

        # Enocde, Decode, Output



    def forward (self, x):
        # Encode, Decode
        pass


if __name__ == "__main__":
    pass