import torch
from torch.utils.data import DataLoader
from dataset import OASISDataset
from modules import ImprovedUNET
import matplotlib.pyplot as plt
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"

images = "OASIS/keras_png_slices_test"
masks  = "OASIS/keras_png_slices_seg_test"
BATCH_SIZE = 1

dataset = OASISDataset(images, masks)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


MODEL_PATH = "checkpoints/improved_unet_epoch1.pth"
model = ImprovedUNET(channels=1, classes=1, bilinear=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


EPSILON = 1e-5
def dice(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersect = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2 * intersect + EPSILON) / (union + EPSILON)).mean().item()


os.makedirs("predicted_masks", exist_ok=True)
dice_scores = []
