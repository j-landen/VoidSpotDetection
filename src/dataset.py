import os
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '**', '*.png'), recursive=True))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # Grayscale
        image_np = np.array(image)  # Convert to numpy for collate
        if self.transform:
            image_np = self.transform(image_np)
        else:
            image_np = np.array(image)
        return image_np, self.image_paths[idx]
