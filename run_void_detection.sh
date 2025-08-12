#!/bin/bash
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=voidspot_detection_new
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# This submission script runs the updated void spot detection pipeline on all
# thermal recordings found under the user's VSA_control directory. It will
# generate binary masks, annotated overlays and a CSV summary for each
# processed video. Modify `IMG_DIR` and `OUT_DIR` according to your project
# layout before launching on the cluster.

# Load modules and activate the conda environment containing OpenCV and
# necessary Python packages. Update the module version if needed for your
# system.
module purge
module load miniconda3/24.3.0
source activate voids

# Define project directories. Adjust these paths to point to your scratch or
# project storage on the HPC system. `SRC_DIR` should point to the location
# of this repository's src directory once deployed to the cluster. `IMG_DIR`
# should point to the root directory containing your .avi files or PNG
# directories. Each subdirectory of `IMG_DIR` (or each AVI file) will be
# processed independently.
PROJECT_DIR=/project/huddlevidmicro/jlanden1
SRC_DIR=$PROJECT_DIR/scripts/voidspot_detection/src
IMG_DIR=$PROJECT_DIR/VSA_control/thermal_vids_output
OUT_DIR=$SRC_DIR/../outputs/new_detector

# Create output directory if it does not exist
mkdir -p "$OUT_DIR"

# Iterate over all AVI files or directories within IMG_DIR. A simple pattern
# match is used here; feel free to adjust the globbing rules for your data.
for path in "$IMG_DIR"/*; do
  name=$(basename "$path")
  echo "[INFO] Processing $name ..."
  # Create a separate output subdirectory per input to avoid naming conflicts
  out_subdir="$OUT_DIR/$name"
  mkdir -p "$out_subdir"
  python "$SRC_DIR/voidspot_detection_pipeline.py" \
    --input "$path" \
    --output_dir "$out_subdir" \
    --method combined \
    --min_area 20 \
    --max_area 2000 \
    --kernel_size 15 \
    --circularity_threshold 0.4 \
    --sigma_small 3.0 \
    --sigma_large 12.0 \
    --use_gpu \
    --multi_scale \
    --otsu_ratio 0.9
done

echo "[DONE] All inputs processed. Outputs saved to $OUT_DIR"