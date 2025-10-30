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
def dice(prediction, target):
    """
    Computes the Dice similarity coefficient between predicted and ground truth masks.
    
    Measures the overlap between two binary masks. Values range
    from 0 to 1 (perfect overlap). 

    Args:
        prediction (torch.Tensor): Raw output from the model (logits), shape [B, C, H, W].
        target (torch.Tensor): Ground truth mask, shape [B, C, H, W].

    Returns:
        float: Average Dice coefficient over the batch.
    """
    # Apply sigmoid to logits to get probabilities
    prediction = torch.sigmoid(prediction)
    # Threshold probabilities at 0.5 to get binary mask
    prediction = (prediction > 0.5).float()
    
    # Ensure target is binary
    target = (target > 0.5).float()
    
    # Compute intersection and union
    intersect = (prediction * target).sum(dim=(1,2,3))
    union = prediction.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    # Compute Dice Coefficient
    # Epsilon used to avoid division by zero
    return ((2 * intersect + EPSILON) / (union + EPSILON)).mean().item()

os.makedirs("predicted_masks", exist_ok=True)
dice_scores = []

with torch.no_grad():
    for i, (image, mask) in enumerate(test_loader):
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        pred_mask = torch.sigmoid(output)
        pred_mask_bin = (pred_mask > 0.5).float()

        plt.imsave(f"predicted_masks/mask_{i}.png", pred_mask_bin[0,0].cpu(), cmap='gray')

        dice_score = dice(output, mask)
        dice_scores.append(dice_score)
        print(f"Image {i}: Dice = {dice_score:.4f}")
