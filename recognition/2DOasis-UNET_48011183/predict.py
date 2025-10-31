import torch
from torch.utils.data import DataLoader
from dataset import OASISDataset
from modules import ImprovedUNET
import matplotlib.pyplot as plt
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"

images = "OASIS/keras_png_slices_test"
masks  = "OASIS/keras_png_slices_seg_test"
# Process one image at a time for prediction
BATCH_SIZE = 1

# Load test dataset and create DataLoader
dataset = OASISDataset(images, masks)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Checkpoints stored in checkpoints/...
MODEL_PATH = "checkpoints/improved_unet_epoch1.pth"
# Initialise model and load saved weights
model = ImprovedUNET(channels=1, classes=1, bilinear=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Evaluation mode
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

# Storing predicted masks
os.makedirs("predicted_masks", exist_ok=True)
dice_scores = []

with torch.no_grad():
    for i, (image, mask) in enumerate(test_loader):
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        # COnvert logits to probabilities
        pred_mask = torch.sigmoid(output)
        # Threshold to get binary mask
        pred_mask_bin = (pred_mask > 0.5).float()

        # Save to disk
        plt.imsave(f"predicted_masks/mask_{i}.png", pred_mask_bin[0,0].cpu(), cmap='gray')

        # Computes dice score for each image, and stored to calculate average
        dice_score = dice(output, mask)
        dice_scores.append(dice_score)
        print(f"Image {i}: Dice = {dice_score:.4f}")

        # Visualise first 5 images, predicted masks, and ground truth
        if i < 5:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            # Input
            axs[0].imshow(image[0,0].cpu(), cmap='gray')
            axs[0].set_title("Input Image")
            axs[0].axis('off')

            # Predicted mask
            axs[1].imshow(pred_mask_bin[0,0].cpu(), cmap='gray')
            axs[1].set_title("Predicted Mask")
            axs[1].axis('off')

            # Ground truth mask
            axs[2].imshow(mask[0,0].cpu(), cmap='gray')
            axs[2].set_title("Ground Truth Mask")
            axs[2].axis('off')

            plt.savefig(f"readme_images/example_{i}.png")

            # Display without blocking execution
            plt.show(block=False)  
            plt.pause(0.5)         
            plt.close()            

# Calculates, prints the overall test set Dice coefficient as metric for model
# performance.
average_dice = sum(dice_scores) / len(dice_scores)
min_dice = min(dice_scores)
max_dice = max(dice_scores)
print(f"\nAverage Test Dice: {average_dice:.4f}")
print(f"\nMin Test Dice: {min_dice:.4f}")
print(f"\nMax Test Dice: {max_dice:.4f}")
