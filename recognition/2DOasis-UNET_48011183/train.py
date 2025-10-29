import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import OASISDataset
from modules import ImprovedUNET

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# Starter values
batch_size = 10
epochs = 10
LR = 0.01
EPSILON = 1e-5

images = "OASIS/keras_png_slices_train"
masks  = "OASIS/keras_png_slices_seg_train"
dataset = OASISDataset(images, masks)

model = ImprovedUNET(channels=1, classes=1, bilinear=True).to(device)
optimiser = torch.optim.Adam(model.parameters(), LR)

criterion = nn.BCEWithLogitsLoss()

# 80/20 Train/Val

train_size = 0.8 * len(dataset)
val_size = 0.2 * len(dataset)

train, val = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train, batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size, shuffle=False)


def dice(prediction, target):
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float()
    
    target = (target > 0.5).float()
    intersect = (prediction * target).sum(dim=(1,2,3))
    union = prediction.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    return ((2 * intersect + EPSILON) / (union + EPSILON)).mean().item()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    dice_c = 0

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        dice_c += dice(outputs, masks)

        if (i + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_dice = dice_c / len(train_loader)

        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_dice += dice(outputs, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Complete: "
          f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_dice:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")