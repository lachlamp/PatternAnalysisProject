import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as tf

class OASISDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 2D OASIS brain MRI images and their corresponding segmentation masks.

    Args:
        image_dir (str): Path to the directory containing MRI slice images.
        mask_dir (str): Path to the directory containing segmentation mask images.
        transform (default: None) Optional function or transform to apply to image and mask tensors (default).

    Returns:
        tuple: (image, mask), where both are torch.FloatTensors of shape [1, H, W].

    Example:
        >>> dataset = OASISDataset("OASIS/keras_png_slices_train", "OASIS/keras_png_slices_seg_train")
        >>> img, mask = dataset[0]
        >>> img.shape, mask.shape
        (torch.Size([1, 256, 256]), torch.Size([1, 256, 256]))
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        # Locations of images
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Forms image and mask pairs
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        # Builds file paths
        img_path = os.path.join(self.image_dir, self.image_files[i])
        mask_path = os.path.join(self.mask_dir, self.mask_files[i])

        # Converts images to single-channel grayscale
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Convert to tensors
        # [1, H, W]
        image = tf.to_tensor(image)    
        mask = tf.to_tensor(mask) 

        # Apply transform (none by default)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
