import os
import cv2
import argparse
from glob import glob

def visualize_and_save(image_dir, mask_dir, overlay_dir, output_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        rel_path = os.path.relpath(img_path, image_dir)
        filename = os.path.basename(img_path)
        subdir = os.path.dirname(rel_path)

        mask_path = os.path.join(mask_dir, subdir, filename.replace(".png", "_mask.png"))
        overlay_path = os.path.join(overlay_dir, subdir, filename.replace(".png", "_overlay.png"))

        if not os.path.exists(mask_path) or not os.path.exists(overlay_path):
            print(f"Skipping {rel_path} — mask or overlay not found.")
            continue

        # Load images
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        overlay = cv2.imread(overlay_path)

        # Resize for consistent display
        h, w = original.shape
        scale = 800 / max(h, w)
        new_size = (int(w * scale), int(h * scale))

        original = cv2.resize(original, new_size)
        mask = cv2.resize(mask, new_size)
        overlay = cv2.resize(overlay, new_size)

        # Stack side-by-side
        stacked = cv2.hconcat([
            cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            overlay
        ])

        # Save result
        save_path = os.path.join(output_dir, subdir, filename.replace(".png", "_compare.png"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, stacked)

    print(f"[DONE] Saved comparisons to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--overlay_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    visualize_and_save(args.image_dir, args.mask_dir, args.overlay_dir, args.output_dir)
