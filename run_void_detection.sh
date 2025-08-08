#!/bin/bash
#SBATCH --account=huddlevidmicro
#SBATCH --job-name=void_detection
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Load modules and activate conda
module purge
module load miniconda3/24.3.0
source activate voids

# Define reusable paths
PROJECT_DIR=/project/huddlevidmicro/jlanden1
SRC_DIR=$PROJECT_DIR/scripts/voidspot_detection/src
IMG_DIR=$PROJECT_DIR/VSA_control/thermal_vids_output
OUT_DIR=$SRC_DIR/../outputs

cd $PROJECT_DIR

echo "[STEP 1] Classical CV pass..."
python $SRC_DIR/classical_voidspot_detector.py \
  --image_dir $IMG_DIR \
  --output_dir $OUT_DIR/classical_pass \
  --batch_size 32

echo "[STEP 2] Unsupervised clustering..."
python $SRC_DIR/unsupervised_detector.py \
  --image_dir $IMG_DIR \
  --output_csv $OUT_DIR/unsupervised_pass/cluster_summary.csv \
  --batch_size 32 \
  --n_clusters 10

echo "[STEP 3] Saving visualizations..."
python $SRC_DIR/visualize_outputs.py \
  --image_dir $IMG_DIR \
  --mask_dir $OUT_DIR/classical_pass/masks \
  --overlay_dir $OUT_DIR/classical_pass/overlays \
  --output_dir $OUT_DIR/comparisons


echo "[STEP 3] Extracting pseudo-labeled outliers..."
python $SRC_DIR/extract_pseudo_labeled.py \
  --summary_csv $OUT_DIR/unsupervised_pass/cluster_summary.csv \
  --image_dir $IMG_DIR \
  --mask_dir $OUT_DIR/classical_pass/masks \
  --output_dir $OUT_DIR/pseudo_labeled

echo "[DONE] Void detection pipeline completed."
