import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import OASISDataset
from modules import ImprovedUNET

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

images = "OASIS/keras_png_slices_train"
masks  = "OASIS/keras_png_slices_seg_train"
dataset = OASISDataset(images, masks)

# Starter values
batch = 10
epochs = 10
LR = 0.01

model = ImprovedUNET(channels=1, classes=1, bilinear=True).to(device)
optimiser = torch.optim.Adam(model.parameters(), LR)

for epoch in range(epochs):
    pass