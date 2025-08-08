import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm

def extract_outliers(summary_csv, image_dir, mask_dir, output_dir):
    df = pd.read_csv(summary_csv)
    outliers = df[df["is_outlier"] == -1]

    img_out_dir = os.path.join(output_dir, "images")
    mask_out_dir = os.path.join(output_dir, "masks")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    print(f"[INFO] Extracting {len(outliers)} outlier frames...")

    for _, row in tqdm(outliers.iterrows(), total=len(outliers)):
        img_path = row["path"]
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Source files
        src_img = img_path
        rel_path = os.path.splitext(os.path.relpath(img_path, image_dir))[0]
        src_mask = os.path.join(mask_dir, f"{rel_path}_mask.png")

        # Destination files
        dst_img = os.path.join(img_out_dir, rel_path + ".png")
        dst_mask = os.path.join(mask_out_dir, rel_path + "_mask.png")
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_mask), exist_ok=True)

        try:
            shutil.copy(src_img, dst_img)
        except Exception as e:
            print(f"[WARNING] Could not copy image: {src_img} — {e}")
            continue

        if os.path.exists(src_mask):
            shutil.copy(src_mask, dst_mask)
        else:
            print(f"[WARNING] Mask not found for {basename}")

    print(f"[DONE] Copied {len(outliers)} images to {output_dir}/images and masks to /masks")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to cluster_summary.csv from Phase 2")
    parser.add_argument("--image_dir", type=str, required=True, help="Base directory where original images are stored")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing classical CV masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted pseudo-labeled data")
    args = parser.parse_args()

    extract_outliers(args.summary_csv, args.image_dir, args.mask_dir, args.output_dir)

if __name__ == "__main__":
    main()
