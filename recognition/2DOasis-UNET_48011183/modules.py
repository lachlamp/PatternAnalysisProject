import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class DoubleConv(nn.Module):
    """
    A building block of the U-Net architecture that applies two consecutive
    convolutional layers, each followed by batch normalisation and ReLU activation.

    Args:
        input_ch (int): Number of input channels.
        output_ch (int): Number of output channels after convolution.

    Forward:
        x (torch.Tensor): Input feature map of shape [B, input_ch, H, W].

    Returns:
        torch.Tensor: Output feature map of shape [B, output_ch, H, W].

    Example:
        >>> x = torch.randn(1, 1, 256, 256)
        >>> block = DoubleConv(1, 64)
        >>> y = block(x)
        >>> y.shape
        torch.Size([1, 64, 256, 256])
    """

    def __init__(self, input_ch, output_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, 3, 1, 1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True),

            # Keeps channels constant
            # Second convolution sequence
            nn.Conv2d(output_ch, output_ch, 3, 1, 1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True)
        )

    # Applies the convolution sequence to tensor
    def forward(self, x):
        return self.conv(x)
    

class Down(nn.Module):
    """
    Downscaling block for U-Net that reduces spatial resolution by a factor of 2
    and increases feature depth. 

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels after convolution.

    Forward:
        x (torch.Tensor): Input feature map of shape [B, in_ch, H, W].

    Returns:
        torch.Tensor: Downsampled feature map of shape [B, out_ch, H/2, W/2].

    Example:
        >>> x = torch.randn(1, 64, 256, 256)
        >>> block = Down(64, 128)
        >>> y = block(x)
        >>> y.shape
        torch.Size([1, 128, 128, 128])
    """

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
    

class Up(nn.Module):
    """
    Upscaling block that restores spatial resolution and fuses encoder
    and decoder features through skip connections.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels after convolution.
        bilinear (default: True): Uses bilinear interpolation for
            upsampling. If False, uses a learnable transposed convolution.

    Forward:
        x (torch.Tensor): Decoder feature map.
        y (torch.Tensor): Corresponding encoder feature map.

    Returns:
        torch.Tensor: Refined feature map of shape [B, out_ch, H, W].

    Example:
        >>> x = torch.randn(1, 256, 64, 64)
        >>> y = torch.randn(1, 128, 128, 128)
        >>> up = Up(384, 128)
        >>> z = up(x, y)
        >>> z.shape
        torch.Size([1, 128, 128, 128])
    """

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

        # Pads x if sizes do not match
        map_x = y.size()[3] - x.size()[3]
        map_y = y.size()[2] - x.size()[2]
        
        x_pad = map_x // 2
        y_pad = map_y // 2

        x = f.pad(x, [x_pad, map_x - x_pad, y_pad, map_y - y_pad])

        # Refines features using DoubleConv
        return self.conv(torch.cat([y, x], 1))


# Reduces feature maps from in_ch to out_ch
class OutConv(nn.Module):
    """
    Final output convolution layer used in UNet architectures. Performs a 1×1
    convolution to map the feature maps from the final decoder layer to the
    desired number of output channels.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels (e.g., number of classes).

    Forward:
        x (Tensor): Input feature map of shape [B, in_ch, H, W].
        
    Returns:
        Tensor: Output map of shape [B, out_ch, H, W].
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class ImprovedUNET(nn.Module):
    """
    Improved UNet architecture for 2D medical image segmentation.

    This model follows the encoder–decoder structure of the standard UNet but
    includes optional bilinear upsampling for smoother feature reconstruction. 
    The network downsamples the input to extract hierarchical features and then 
    upsamples while concatenating corresponding encoder features to recover 
    spatial detail.

    Args:
        channels (int): Number of input channels (1 for grayscale MRI slices).
        classes (int): Number of output segmentation classes.
        bilinear (bool): If True, uses bilinear interpolation for upsampling instead
        of transposed convolutions.

    Attributes:
        inconv (DoubleConv): Initial convolution block for feature extraction.
        down1–down4 (Down): Encoder layers for reducing spatial size.
        up1–up4 (Up): Decoder layers for feature upsampling.
        outconv (OutConv): 1×1 convolution mapping to output segmentation map.

    Forward:
        x (Tensor): Input image tensor of shape [B, channels, H, W].

    Returns:
        Tensor: Segmentation logits of shape [B, classes, H, W].
    """

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
        # Doubling the channel count enables high-level features to be extracted
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        self.down4 = Down(512, 1024 // self.factor)
    
    def decode(self):
        # Reverses the steps of the encoder
        self.up1 = Up(1024 // self.factor + 512, 512 // self.factor, self.bilinear)
        self.up2 = Up(512 // self.factor + 256, 256 // self.factor, self.bilinear)
        self.up3 = Up(256 // self.factor + 128, 128 // self.factor, self.bilinear)
        self.up4 = Up(128 // self.factor + 64, 64, self.bilinear)

    def forward (self, x):
        # Feature extraction from input image
        x1 = self.inconv(x)

        # Encoder path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final 1×1 convolution
        return self.outconv(x)