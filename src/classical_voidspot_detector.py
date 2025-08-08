import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from dataset import FrameDataset
from torch.utils.data import DataLoader
from skimage import morphology

def detect_void_spots(image, min_area=50):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological clean-up
    cleaned = morphology.remove_small_objects(thresh > 0, min_size=min_area)
    mask = cleaned.astype(np.uint8) * 255

    return mask

def main(image_dir, output_dir, batch_size=16):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    dataset = FrameDataset(image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print(f"[INFO] Loaded {len(dataset)} frames from {image_dir}")

    for batch in tqdm(loader):
        images, paths = batch
        for img_tensor, path in zip(images, paths):
            img_np = np.array(img_tensor)

            mask = detect_void_spots(img_np)
            overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            overlay[mask > 0] = [0, 0, 255]  # Mark detected areas in red

            rel_path = os.path.relpath(path, image_dir)
            save_base = os.path.splitext(rel_path)[0]

            mask_path = os.path.join(output_dir, "masks", f"{save_base}_mask.png")
            overlay_path = os.path.join(output_dir, "overlays", f"{save_base}_overlay.png")

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)

            cv2.imwrite(mask_path, mask)
            cv2.imwrite(overlay_path, overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing PNG frames")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save masks and overlays")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    main(args.image_dir, args.output_dir, args.batch_size)
