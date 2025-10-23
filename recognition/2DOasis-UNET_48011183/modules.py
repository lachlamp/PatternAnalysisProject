import torch
import torch.nn as nn
import torch.nn.functional as f

class DoubleConv(nn.Module):
    def __init__(self, input_ch, output_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, 3, 1, 1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True),

            nn.Conv2d(output_ch, output_ch, 3, 1, 1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)




if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    block = DoubleConv(1, 64)
    y = block(x)
    print(y.shape)