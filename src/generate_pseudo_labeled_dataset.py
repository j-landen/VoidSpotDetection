import os
import shutil
import pandas as pd
import cv2
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------

CLUSTER_CSV = "Z:/bedfordlab/Thermal_processing_Jason/VoidSpotDetection/outputs/unsupervised_pass/cluster_summary.csv"
MASK_ROOT = "Z:/bedfordlab/Thermal_processing_Jason/VoidSpotDetection/outputs/classical_pass/masks"
IMAGE_ROOT = "Z:/bedfordlab/Thermal_processing_Jason/vsa_thermal_output"  # Adjust to full path if needed
OUTPUT_IMAGE_DIR = "Z:/bedfordlab/Thermal_processing_Jason/VoidSpotDetection/outputs/filtered/images"
OUTPUT_MASK_DIR = "Z:/bedfordlab/Thermal_processing_Jason/VoidSpotDetection/outputs/filtered/masks"
MIN_MASK_AREA = 100  # Set to 0 to include all outliers
MAX_MASK_AREA = 5200

# ----------------------------
# Utilities
# ----------------------------

def get_mask_area(mask_path):
    if not os.path.exists(mask_path):
        return 0
    mask = cv2.imread(mask_path, 0)
    return (mask > 0).sum()

def create_output_path(base_dir, relative_path):
    return os.path.join(base_dir, *relative_path.split("/"))

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ----------------------------
# Main logic
# ----------------------------

def main():
    df = pd.read_csv(CLUSTER_CSV)
    print(f"[INFO] Loaded {len(df)} total frames from {CLUSTER_CSV}")

    df_outliers = df[df["is_outlier"] == -1].copy()
    print(f"[INFO] Found {len(df_outliers)} frames flagged as outliers")

    kept = 0
    for _, row in tqdm(df_outliers.iterrows(), total=len(df_outliers), desc="Processing outliers"):
        rel_path = row["path"].replace("\\", "/")  # ensure UNIX-style paths
        image_path = os.path.join(IMAGE_ROOT, rel_path)
        mask_path = os.path.join(MASK_ROOT, rel_path.replace("frame", "frame").replace(".png", "_mask.png"))

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        area = get_mask_area(mask_path)
        if area < MIN_MASK_AREA or area > MAX_MASK_AREA:
            continue

        # Generate output paths
        out_img_path = create_output_path(OUTPUT_IMAGE_DIR, rel_path)
        out_mask_path = create_output_path(OUTPUT_MASK_DIR, rel_path.replace("frame", "frame").replace(".png", "_mask.png"))

        ensure_dir(out_img_path)
        ensure_dir(out_mask_path)

        shutil.copy2(image_path, out_img_path)
        shutil.copy2(mask_path, out_mask_path)

        kept += 1

    print(f"[DONE] Copied {kept} outlier frames and masks to pseudo_labeled/")

if __name__ == "__main__":
    main()
