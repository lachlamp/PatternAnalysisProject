import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import OASISDataset
from modules import ImprovedUNET
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device.upper()}")

# Training parameters:
batch_size = 8
epochs = 1
LR = 0.01 # Learning rate for the optimiser
EPSILON = 1e-5

images = "OASIS/keras_png_slices_train"
masks  = "OASIS/keras_png_slices_seg_train"
dataset = OASISDataset(images, masks)

# Instantiate the Improved UNet model
model = ImprovedUNET(channels=1, classes=1, bilinear=True).to(device)

# Adam Optimiser for updating model parameters during training
optimiser = torch.optim.Adam(model.parameters(), LR)

# Loss function: Binary Cross-Entropy with logits
criterion = nn.BCEWithLogitsLoss()

# Split dataset into 80% training and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Dataloaders to iterate over training and validation data
train, val = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train, batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size, shuffle=False)

# Metrics for plotting
train_losses, val_losses = [], []
train_dices, val_dices = [], []

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

# Checkpoints in case training does not finish
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(epochs):
    # Set model to training mode
    model.train()
    train_loss = 0
    dice_c = 0

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Resets gradients
        optimiser.zero_grad()
        # Forward pass through the model
        outputs = model(images)

        # Compute loss against ground truth
        loss = criterion(outputs, masks)
        # Backpropagate gradients
        loss.backward()
        # Update model parameters
        optimiser.step()

        # Accumulate loss and dice coefficient
        train_loss += loss.item()
        dice_c += dice(outputs, masks)

        # Progress Update every 20 batches
        if (i + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
    avg_train_loss = train_loss / len(train_loader)
    avg_dice = dice_c / len(train_loader)

    # Sets model to evaluation mode
    model.eval()
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        # Iterate over validation dataset
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            val_dice += dice(outputs, masks)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    # Metrics for plotting
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_dices.append(avg_dice)
    val_dices.append(avg_val_dice)

    # Epoch summary
    print(f"Epoch [{epoch+1}/{epochs}] Complete: "
        f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_dice:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
    
    # Save model checkpoint for this epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"improved_unet_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")