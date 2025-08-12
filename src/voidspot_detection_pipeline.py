"""
Detection and analysis pipeline for void (urination) events in thermal videos.

This module implements an end‑to‑end workflow to segment cooler "void spots" in
thermal imagery and summarise them frame by frame. It supports processing
either a directory of pre‑extracted PNG frames or a raw AVI video. The
extracted segmentation masks, annotated frames and a per‑frame CSV summary are
persisted to disk.

Key features of this pipeline:

* **Colour‑based segmentation:** Thermal videos in this application encode
  temperature as colour. Cooler regions (urine pools) tend to appear as
  blue/purple patches whereas the mouse appears warmer (orange/red). We
  therefore convert each frame to the HSV colour space and threshold on the
  hue channel to isolate candidate cold spots. A two‑stage morphological
  cleaning removes spurious noise and small artefacts.

* **Connected component analysis:** Each contiguous region of the binary
  segmentation mask is treated as a single candidate void spot. We filter out
  very small components using an area threshold to minimise false positives.

* **Per‑spot metrics:** For every retained component we compute its pixel
  area and centroid coordinates. These metrics, along with the number of
  detected spots, are written to a CSV summary. Centroids are reported in
  image coordinates with the origin in the top‑left corner.

* **Flexible inputs:** The `--input` argument accepts either an `.avi` video
  file or a directory containing PNG frames. When a video is supplied
  `cv2.VideoCapture` is used to iterate through frames without saving them to
  disk first. When a directory is supplied the script recurses through
  subdirectories to process every `.png` file.

* **Visual QA outputs:** For each processed frame we save two visualisations:
  a binary mask image highlighting segmented cold spots and an annotated copy
  of the original frame with red bounding circles drawn around each detected
  component. These overlays help debug the chosen parameters without having to
  manually inspect raw numeric outputs.

Example usage:

```bash
python voidspot_detection_pipeline.py \
    --input /project/huddlevidmicro/jlanden1/VSA_control/thermal_vids_output/Round\ 1/vid001.avi \
    --output_dir /project/huddlevidmicro/jlanden1/scripts/voidspot_detection/outputs/new_detector \
    --min_area 200 --hue_min 90 --hue_max 160
```

The above command will analyse a single video and save all outputs into the
specified directory. You may adjust `min_area`, `hue_min` and `hue_max`
depending on the colour palette of your recordings.

Author: ChatGPT Agent
"""

import argparse
import csv
import os
from glob import glob
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def detect_void_spots(
    frame: np.ndarray,
    *,
    method: str = "morph",  # 'color', 'morph', 'dog', or 'combined'
    hue_min: int = 90,
    hue_max: int = 160,
    sat_min: int = 50,
    val_min: int = 50,
    min_area: int = 20,
    max_area: int = 2000,
    kernel_size: int = 15,
    circularity_threshold: float = 0.4,
    use_gpu: bool = False,
    sigma_small: float = 3.0,
    sigma_large: float = 12.0,
    exclude_hot: bool = True,
    hot_val_threshold: int = 200,
    hot_area_threshold: int = 5000,
    otsu_ratio: float = 1.0,
    multi_scale: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """
    Segment candidate void spots from a colour frame using either colour masking or
    morphological filters.

    The colour-based method (method='color') retains the previous behaviour of
    thresholding on HSV hue/saturation/value to isolate blue/purple regions. The
    morphological method (method='morph') uses top-hat and black-hat transforms
    to detect both bright (fresh, warm urine) and dark (cool, evaporated) spots
    relative to the local background, then filters connected components by
    size and circularity to discard the mouse and other artefacts.

    Parameters
    ----------
    frame : np.ndarray
        An image in BGR format.
    method : str
        Detection algorithm to use. Supported options are:

        * `'color'` – apply a fixed HSV threshold to isolate cold (blue/purple) regions.
        * `'morph'` – extract small bright/dark details using white/black top‑hat
          transforms. This emphasises both cool spots and freshly deposited, warmer
          urine that appear as bright blobs【871308506922065†L118-L169】.
        * `'dog'` – use a difference‑of‑Gaussians (DoG) filter to highlight local
          intensity extrema (both hot and cold spots). The filter subtracts a
          heavily blurred version of the frame from a lightly blurred version,
          boosting blob‑like structures while suppressing larger scale
          background variations. The standard deviations of the two Gaussians
          are controlled via `sigma_small` and `sigma_large`.
        * `'combined'` – sum the responses of the morphological (top‑hat) and DoG
          filters to leverage both spatial and colour information. This tends to
          be the most robust option across varying palettes and spot
          temperatures.
    hue_min, hue_max, sat_min, val_min : int
        Parameters for the colour-based detector. Ignored when method='morph'.
    min_area : int
        Minimum connected-component area to keep (pixels).
    max_area : int
        Maximum connected-component area to keep. This helps remove the mouse
        and other large objects. Set to a large number to disable.
    kernel_size : int
        Diameter of the structuring element used in morphological opening and
        closing (for method 'morph' or 'combined'). It should approximate the
        largest void spot diameter in pixels. For the DoG‑based detector the
        kernel size determines the extent of morphological operations but is
        largely superseded by `sigma_small` and `sigma_large`.
    circularity_threshold : float
        Minimum circularity (4π·area/perimeter²) to keep a component. Circular
        void spots will have values close to 1, while elongated mice or
        artefacts yield lower values.
    use_gpu : bool
        If True and PyTorch is installed with CUDA support, perform the
        morphological filtering and DoG computation on the GPU. Connected
        component calculation still occurs on the CPU.

    sigma_small : float
        Standard deviation of the smaller Gaussian in the DoG filter (in
        pixels). Smaller values highlight finer details. Only relevant for
        `method` in {'dog', 'combined'}.

    sigma_large : float
        Standard deviation of the larger Gaussian in the DoG filter (in
        pixels). Larger values control the scale of background suppression.
        Only relevant for `method` in {'dog', 'combined'}.

    Returns
    -------
    mask : np.ndarray
        Binary mask of detected void spots. Pixels belonging to retained
        components are set to 255 and all others to 0.
    components : list of tuples
        A list of `(area, cx, cy)` entries for each valid spot, where
        `area` is the pixel area, and `cx`, `cy` are the centroid
        coordinates (floating point) relative to the original frame.
    """
    # If the user requests the legacy colour-based method, fall back to the old
    # implementation. This branch is retained for backwards compatibility and
    # cases where the thermal palette uses a fixed colour map.
    # The parameters `exclude_hot`, `hot_val_threshold` and `hot_area_threshold` control
    # a safeguard to remove detections inside large hot regions (e.g. the mouse body).
    # See function signature for details.
    method = method.lower()
    if method == "color":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if hue_min <= hue_max:
            hue_mask = cv2.inRange(h, hue_min, hue_max)
        else:
            lower_part = cv2.inRange(h, hue_min, 179)
            upper_part = cv2.inRange(h, 0, hue_max)
            hue_mask = cv2.bitwise_or(lower_part, upper_part)
        sat_mask = cv2.inRange(s, sat_min, 255)
        val_mask = cv2.inRange(v, val_min, 255)
        combined = cv2.bitwise_and(hue_mask, cv2.bitwise_and(sat_mask, val_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        final_mask = np.zeros_like(cleaned, dtype=np.uint8)
        components: List[Tuple[int, float, float]] = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or (max_area and area > max_area):
                continue
            cx, cy = centroids[i]
            components.append((int(area), float(cx), float(cy)))
            final_mask[labels == i] = 255
        return final_mask, components

    # For all other methods we work on a single channel intensity image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use a small Gaussian blur to suppress sensor noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute filter responses based on the chosen method. We build up
    # `response_np` as a uint8 image highlighting candidate void spots.
    response_np: np.ndarray
    if method == "morph":
        # Only morphological top/bottom hat. Optionally accumulate responses over
        # multiple structuring element sizes (multi_scale=True) to capture both
        # small and large void spots. The structuring sizes are derived from
        # `kernel_size` by halving and doubling. Each scale uses an odd
        # diameter. The final response is the average of the per‑scale
        # responses.
        scales: List[int]
        if multi_scale:
            scales = []
            # Use up to three scales: half, base and double kernel_size
            for factor in [0.5, 1.0, 2.0]:
                ks = int(round(kernel_size * factor))
                if ks < 3:
                    ks = 3
                # Ensure odd
                if ks % 2 == 0:
                    ks += 1
                scales.append(ks)
            # Remove duplicates while preserving order
            seen = set()
            scales = [s for s in scales if not (s in seen or seen.add(s))]
        else:
            ks = kernel_size
            if ks < 3:
                ks = 3
            if ks % 2 == 0:
                ks += 1
            scales = [ks]

        if use_gpu:
            try:
                import torch
                import torch.nn.functional as F

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img_tensor = torch.from_numpy(blur.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                combined_resp = None
                for s in scales:
                    pad = s // 2
                    # Opening: erosion then dilation via max_pool2d
                    eroded = -F.max_pool2d(-img_tensor, s, stride=1, padding=pad)
                    opened = F.max_pool2d(eroded, s, stride=1, padding=pad)
                    # Top‑hat and black‑hat
                    top_hat = img_tensor - opened
                    dilated = F.max_pool2d(img_tensor, s, stride=1, padding=pad)
                    closed = -F.max_pool2d(-dilated, s, stride=1, padding=pad)
                    black_hat = closed - img_tensor
                    resp = top_hat + black_hat
                    if combined_resp is None:
                        combined_resp = resp
                    else:
                        combined_resp = combined_resp + resp
                combined_resp = combined_resp / float(len(scales))
                response_np = (combined_resp.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
            except Exception:
                use_gpu = False
        if not use_gpu:
            # CPU implementation
            combined_resp = np.zeros_like(blur, dtype=np.float32)
            for s in scales:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
                opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, se)
                closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, se)
                top_hat = cv2.subtract(blur, opening).astype(np.float32)
                black_hat = cv2.subtract(closing, blur).astype(np.float32)
                combined_resp += top_hat + black_hat
            combined_resp /= float(len(scales))
            # Normalise to 0–255
            resp_min = combined_resp.min()
            resp_max = combined_resp.max()
            if (resp_max - resp_min) > 1e-6:
                response_np = ((combined_resp - resp_min) / (resp_max - resp_min) * 255.0).astype(np.uint8)
            else:
                response_np = np.zeros_like(combined_resp, dtype=np.uint8)
    elif method == "dog":
        # Difference‑of‑Gaussians filtering to highlight local extrema
        if use_gpu:
            try:
                import torch
                import torch.nn.functional as F
                from math import exp

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Prepare image as 1×1×H×W tensor in [0,1]
                img_tensor = torch.from_numpy(blur.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

                def gaussian_kernel(sigma: float) -> torch.Tensor:
                    # Compute a 1D Gaussian kernel. Use size 6*sigma to cover support, odd length.
                    kernel_size = max(3, int(2 * round(3 * sigma) + 1))
                    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
                    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                    kernel /= kernel.sum()
                    return kernel

                # Build separable Gaussian kernels for both sigmas
                k_small = gaussian_kernel(sigma_small)
                k_large = gaussian_kernel(sigma_large)
                # Reshape for conv2d (out_channels,in_channels,KH,KW)
                k_small_2d = k_small.unsqueeze(1) @ k_small.unsqueeze(0)
                k_large_2d = k_large.unsqueeze(1) @ k_large.unsqueeze(0)
                k_small_2d = k_small_2d.unsqueeze(0).unsqueeze(0)
                k_large_2d = k_large_2d.unsqueeze(0).unsqueeze(0)
                # Convolve using padding to preserve size
                # group=1 because image has one channel
                small_blur = F.conv2d(img_tensor, k_small_2d, padding=k_small_2d.shape[-1]//2)
                large_blur = F.conv2d(img_tensor, k_large_2d, padding=k_large_2d.shape[-1]//2)
                dog = small_blur - large_blur
                dog_abs = dog.abs().squeeze()
                # Normalise to 0–1 then scale to 0–255
                dog_min = dog_abs.min()
                dog_max = dog_abs.max()
                if (dog_max - dog_min) > 1e-6:
                    dog_norm = (dog_abs - dog_min) / (dog_max - dog_min)
                else:
                    dog_norm = torch.zeros_like(dog_abs)
                response_np = (dog_norm * 255.0).cpu().numpy().astype(np.uint8)
            except Exception:
                use_gpu = False
        if not use_gpu:
            # CPU implementation using OpenCV Gaussians
            small_blur = cv2.GaussianBlur(blur, (0, 0), sigma_small)
            large_blur = cv2.GaussianBlur(blur, (0, 0), sigma_large)
            dog = small_blur.astype(np.float32) - large_blur.astype(np.float32)
            dog_abs = np.abs(dog)
            dog_min = dog_abs.min()
            dog_max = dog_abs.max()
            if (dog_max - dog_min) > 1e-6:
                dog_norm = (dog_abs - dog_min) / (dog_max - dog_min)
            else:
                dog_norm = np.zeros_like(dog_abs)
            response_np = (dog_norm * 255.0).astype(np.uint8)
    elif method == "combined":
        # Combine morphological top/bottom hat with DoG
        # Compute morphological response first (optionally multi-scale)
        scales: List[int]
        if multi_scale:
            scales = []
            for factor in [0.5, 1.0, 2.0]:
                ks = int(round(kernel_size * factor))
                if ks < 3:
                    ks = 3
                if ks % 2 == 0:
                    ks += 1
                scales.append(ks)
            seen = set()
            scales = [s for s in scales if not (s in seen or seen.add(s))]
        else:
            ks = kernel_size
            if ks < 3:
                ks = 3
            if ks % 2 == 0:
                ks += 1
            scales = [ks]
        morph_np = None
        if use_gpu:
            try:
                import torch
                import torch.nn.functional as F

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img_tensor = torch.from_numpy(blur.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                combined_resp = None
                for s in scales:
                    pad = s // 2
                    eroded = -F.max_pool2d(-img_tensor, s, stride=1, padding=pad)
                    opened = F.max_pool2d(eroded, s, stride=1, padding=pad)
                    top_hat = img_tensor - opened
                    dilated = F.max_pool2d(img_tensor, s, stride=1, padding=pad)
                    closed = -F.max_pool2d(-dilated, s, stride=1, padding=pad)
                    black_hat = closed - img_tensor
                    resp = top_hat + black_hat
                    if combined_resp is None:
                        combined_resp = resp
                    else:
                        combined_resp = combined_resp + resp
                combined_resp = combined_resp / float(len(scales))
                morph_np = (combined_resp.squeeze().cpu().numpy() * 255.0).astype(np.float32)
            except Exception:
                use_gpu = False
        if not use_gpu:
            combined_resp = np.zeros_like(blur, dtype=np.float32)
            for s in scales:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
                opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, se)
                closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, se)
                top_hat = cv2.subtract(blur, opening).astype(np.float32)
                black_hat = cv2.subtract(closing, blur).astype(np.float32)
                combined_resp += top_hat + black_hat
            combined_resp /= float(len(scales))
            # Normalise to 0–255
            resp_min = combined_resp.min()
            resp_max = combined_resp.max()
            if (resp_max - resp_min) > 1e-6:
                morph_np = ((combined_resp - resp_min) / (resp_max - resp_min) * 255.0).astype(np.float32)
            else:
                morph_np = np.zeros_like(combined_resp, dtype=np.float32)
        # Compute DoG response
        if use_gpu:
            try:
                import torch
                import torch.nn.functional as F

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img_tensor = torch.from_numpy(blur.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                def gaussian_kernel(sigma: float) -> torch.Tensor:
                    size = max(3, int(2 * round(3 * sigma) + 1))
                    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
                    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                    kernel /= kernel.sum()
                    return kernel
                k_small = gaussian_kernel(sigma_small)
                k_large = gaussian_kernel(sigma_large)
                k_small_2d = k_small.unsqueeze(1) @ k_small.unsqueeze(0)
                k_large_2d = k_large.unsqueeze(1) @ k_large.unsqueeze(0)
                k_small_2d = k_small_2d.unsqueeze(0).unsqueeze(0)
                k_large_2d = k_large_2d.unsqueeze(0).unsqueeze(0)
                small_blur = F.conv2d(img_tensor, k_small_2d, padding=k_small_2d.shape[-1]//2)
                large_blur = F.conv2d(img_tensor, k_large_2d, padding=k_large_2d.shape[-1]//2)
                dog = small_blur - large_blur
                dog_abs = dog.abs().squeeze()
                dog_min = dog_abs.min()
                dog_max = dog_abs.max()
                if (dog_max - dog_min) > 1e-6:
                    dog_norm = (dog_abs - dog_min) / (dog_max - dog_min)
                else:
                    dog_norm = torch.zeros_like(dog_abs)
                dog_np = (dog_norm * 255.0).cpu().numpy().astype(np.float32)
            except Exception:
                use_gpu = False
        if not use_gpu:
            small_blur = cv2.GaussianBlur(blur, (0, 0), sigma_small)
            large_blur = cv2.GaussianBlur(blur, (0, 0), sigma_large)
            dog = small_blur.astype(np.float32) - large_blur.astype(np.float32)
            dog_abs = np.abs(dog)
            dog_min = dog_abs.min()
            dog_max = dog_abs.max()
            if (dog_max - dog_min) > 1e-6:
                dog_norm = (dog_abs - dog_min) / (dog_max - dog_min)
            else:
                dog_norm = np.zeros_like(dog_abs)
            dog_np = (dog_norm * 255.0).astype(np.float32)
        # Combine the two responses. We use a simple weighted sum; equal weights by default.
        combined = morph_np + dog_np
        # Normalise the result to 0–255 to standardise thresholding
        resp_min = combined.min()
        resp_max = combined.max()
        if (resp_max - resp_min) > 1e-6:
            response_np = ((combined - resp_min) / (resp_max - resp_min) * 255.0).astype(np.uint8)
        else:
            response_np = np.zeros_like(combined, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported detection method: {method}")

    # Optionally suppress detections inside large hot regions (the mouse)
    # Compute a mask of high‑value (hot) areas and zero the response there. This
    # helps avoid false positives around the warm mouse body. We first find
    # connected components in the high‑value mask and only keep those larger
    # than `hot_area_threshold` pixels, treating them as the mouse. Smaller
    # bright spots (fresh urine) are retained.【871308506922065†L118-L169】
    if exclude_hot:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # threshold on the V (brightness) channel to identify hot areas
        v_channel = hsv_frame[:, :, 2]
        hot_mask = cv2.inRange(v_channel, hot_val_threshold, 255)
        # Remove isolated hot pixels with a morphological opening
        se_hot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, se_hot)
        # Find connected components in the hot mask
        num_hot_labels, hot_labels, hot_stats, _ = cv2.connectedComponentsWithStats(hot_mask, connectivity=8)
        large_hot = np.zeros_like(hot_mask, dtype=np.uint8)
        for i in range(1, num_hot_labels):
            area = hot_stats[i, cv2.CC_STAT_AREA]
            if area >= hot_area_threshold:
                large_hot[hot_labels == i] = 255
        if np.any(large_hot):
            # Zero the filter response inside large hot regions
            response_np = cv2.bitwise_and(response_np, cv2.bitwise_not(large_hot))

    # Threshold the response image to obtain a binary mask. We first compute the
    # Otsu threshold and then optionally scale it down using `otsu_ratio` to
    # increase sensitivity. A ratio < 1 lowers the threshold, allowing weaker
    # response values to be kept. This can be helpful when void spots produce
    # subtle signals relative to the background. For highly varying frames you
    # may wish to adjust or disable this behaviour.
    otsu_ret, _ = cv2.threshold(response_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scaled_thresh_value = float(otsu_ret) * otsu_ratio
    _, thresh = cv2.threshold(response_np, scaled_thresh_value, 255, cv2.THRESH_BINARY)
    # Remove small noise with a small opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

    # Connected components analysis to keep only sufficiently circular regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_mask = np.zeros_like(cleaned, dtype=np.uint8)
    components: List[Tuple[int, float, float]] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or (max_area and area > max_area):
            continue
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < circularity_threshold:
            continue
        final_mask[labels == i] = 255
        cx, cy = centroids[i]
        components.append((int(area), float(cx), float(cy)))

    return final_mask, components


def annotate_frame(frame: np.ndarray, components: List[Tuple[int, float, float]]) -> np.ndarray:
    """
    Draw bounding circles around detected components on a copy of the input frame.

    Each circle's radius is derived from the component area assuming a circular
    shape: `radius = sqrt(area / pi)`. Circles are drawn in red. A small
    crosshair is also drawn at each centroid.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame on which to draw annotations. The frame is modified
        in place.
    components : list of tuples
        Components returned by `detect_void_spots` in the form `(area, cx, cy)`.

    Returns
    -------
    annotated : np.ndarray
        The annotated BGR frame.
    """
    annotated = frame.copy()
    for area, cx, cy in components:
        # Compute radius assuming circular shape
        radius = int(np.sqrt(area / np.pi))
        center = (int(round(cx)), int(round(cy)))
        # Draw circle
        cv2.circle(annotated, center, radius, (0, 0, 255), 2)  # red
        # Draw crosshair at centroid
        cv2.drawMarker(annotated, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)
    return annotated


from tqdm import tqdm


def process_png_frames(
    frame_paths: Iterable[str],
    video_id: str,
    output_mask_dir: str,
    output_overlay_dir: str,
    csv_writer: csv.writer,
    *,
    method: str,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    val_min: int,
    min_area: int,
    max_area: int,
    kernel_size: int,
    circularity_threshold: float,
    use_gpu: bool,
    sigma_small: float,
    sigma_large: float,
    exclude_hot: bool,
    hot_val_threshold: int,
    hot_area_threshold: int,
    otsu_ratio: float,
    multi_scale: bool,
):
    """
    Process a sequence of PNG frames and write results to CSV.

    Parameters
    ----------
    frame_paths : iterable of str
        Paths to individual PNG files sorted in ascending order.
    video_id : str
        Identifier associated with this collection of frames (e.g. directory or video name).
    output_mask_dir : str
        Directory where binary masks will be saved.
    output_overlay_dir : str
        Directory where annotated frames will be saved.
    csv_writer : csv.writer
        Open CSV writer to append per‑frame results.
    hue_min, hue_max, sat_min, val_min, min_area
        Threshold parameters forwarded to `detect_void_spots`.
    """
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=f"{video_id}", unit="frame")):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        mask, components = detect_void_spots(
            frame,
            method=method,
            hue_min=hue_min,
            hue_max=hue_max,
            sat_min=sat_min,
            val_min=val_min,
            min_area=min_area,
            max_area=max_area,
            kernel_size=kernel_size,
            circularity_threshold=circularity_threshold,
            use_gpu=use_gpu,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            exclude_hot=exclude_hot,
            hot_val_threshold=hot_val_threshold,
            hot_area_threshold=hot_area_threshold,
            otsu_ratio=otsu_ratio,
            multi_scale=multi_scale,
        )
        overlay = annotate_frame(frame, components)
        rel_name = os.path.basename(frame_path)
        name_base, _ = os.path.splitext(rel_name)
        mask_path = os.path.join(output_mask_dir, f"{name_base}_mask.png")
        overlay_path = os.path.join(output_overlay_dir, f"{name_base}_overlay.png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)
        sizes = ";".join(str(comp[0]) for comp in components)
        xs = ";".join(f"{comp[1]:.1f}" for comp in components)
        ys = ";".join(f"{comp[2]:.1f}" for comp in components)
        csv_writer.writerow([video_id, frame_idx, len(components), sizes, xs, ys])


def process_video(
    video_path: str,
    output_mask_dir: str,
    output_overlay_dir: str,
    csv_writer: csv.writer,
    *,
    method: str,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    val_min: int,
    min_area: int,
    max_area: int,
    kernel_size: int,
    circularity_threshold: float,
    use_gpu: bool,
    sigma_small: float,
    sigma_large: float,
    exclude_hot: bool,
    hot_val_threshold: int,
    hot_area_threshold: int,
    otsu_ratio: float,
    multi_scale: bool,
):
    """
    Process frames from an AVI video file.

    Parameters
    ----------
    video_path : str
        Path to a video file supported by OpenCV.
    output_mask_dir : str
        Directory where binary masks will be saved.
    output_overlay_dir : str
        Directory where annotated frames will be saved.
    csv_writer : csv.writer
        Open CSV writer to append per‑frame results.
    hue_min, hue_max, sat_min, val_min, min_area
        Threshold parameters forwarded to `detect_void_spots`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_iter = range(total_frames) if total_frames > 0 else iter(int, 1)
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=video_id, unit="frame")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask, components = detect_void_spots(
            frame,
            method=method,
            hue_min=hue_min,
            hue_max=hue_max,
            sat_min=sat_min,
            val_min=val_min,
            min_area=min_area,
            max_area=max_area,
            kernel_size=kernel_size,
            circularity_threshold=circularity_threshold,
            use_gpu=use_gpu,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            exclude_hot=exclude_hot,
            hot_val_threshold=hot_val_threshold,
            hot_area_threshold=hot_area_threshold,
            otsu_ratio=otsu_ratio,
            multi_scale=multi_scale,
        )
        overlay = annotate_frame(frame, components)
        name_base = f"frame{frame_idx:06d}"
        mask_path = os.path.join(output_mask_dir, f"{name_base}_mask.png")
        overlay_path = os.path.join(output_overlay_dir, f"{name_base}_overlay.png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)
        sizes = ";".join(str(comp[0]) for comp in components)
        xs = ";".join(f"{comp[1]:.1f}" for comp in components)
        ys = ";".join(f"{comp[2]:.1f}" for comp in components)
        csv_writer.writerow([video_id, frame_idx, len(components), sizes, xs, ys])
        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()


def gather_png_frames(input_dir: str) -> Tuple[str, List[str]]:
    """
    Recursively collect all PNG files under `input_dir`. Infer a video/clip
    identifier from the directory name.

    Returns the base directory name (video_id) and a sorted list of frame paths.
    """
    video_id = os.path.basename(os.path.normpath(input_dir))
    frame_paths = sorted(glob(os.path.join(input_dir, "**", "*.png"), recursive=True))
    return video_id, frame_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and quantify void spots from thermal videos or frames.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an AVI file or a directory of PNG frames. The script will automatically detect the type.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where masks, overlays and CSV will be saved.",
    )
    parser.add_argument("--min_area", type=int, default=20, help="Minimum area of detected spot to keep (pixels)")
    parser.add_argument(
        "--max_area",
        type=int,
        default=2000,
        help="Maximum area of detected spot to keep. Components larger than this are discarded as likely mice or artefacts.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="combined",
        choices=["color", "morph", "dog", "combined"],
        help="Detection method: choose from 'color' (HSV threshold), 'morph' (top/bottom‑hat), 'dog' (difference‑of‑Gaussians) or 'combined' (sum of morph and DoG). 'combined' is recommended for robust detection.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=15,
        help="Structuring element size for morphological opening/closing (odd integer)",
    )
    parser.add_argument(
        "--circularity_threshold",
        type=float,
        default=0.4,
        help="Minimum circularity (0–1) required for a component to be considered a void spot", 
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="If set, use PyTorch on the GPU for morphological operations if available.",
    )

    parser.add_argument(
        "--sigma_small",
        type=float,
        default=3.0,
        help="Standard deviation of the smaller Gaussian for DoG detection (pixels). Only used when method is 'dog' or 'combined'.",
    )
    parser.add_argument(
        "--sigma_large",
        type=float,
        default=12.0,
        help="Standard deviation of the larger Gaussian for DoG detection (pixels). Only used when method is 'dog' or 'combined'.",
    )

    parser.add_argument(
        "--no_exclude_hot",
        action="store_true",
        help="Disable suppression of detections inside large hot regions (mouse body). By default hot areas are excluded.",
    )
    parser.add_argument(
        "--hot_val_threshold",
        type=int,
        default=200,
        help="Brightness threshold (0–255) for hot region segmentation. Only applies when not using --no_exclude_hot.",
    )
    parser.add_argument(
        "--hot_area_threshold",
        type=int,
        default=5000,
        help="Minimum area (pixels) for a connected hot region to be considered the mouse. Used to exclude the mouse body.",
    )

    parser.add_argument(
        "--otsu_ratio",
        type=float,
        default=1.0,
        help="Multiply Otsu's threshold by this factor (<1 lowers the threshold, increasing sensitivity).",
    )

    parser.add_argument(
        "--multi_scale",
        action="store_true",
        help="Enable multi-scale morphological filtering. This runs the top-/bottom‑hat filter at half, base and double the kernel size and averages the responses, capturing spots of different sizes.",
    )
    parser.add_argument(
        "--hue_min",
        type=int,
        default=90,
        help="Lower bound of hue (0–179) to detect cooler colours (default 90)",
    )
    parser.add_argument(
        "--hue_max",
        type=int,
        default=160,
        help="Upper bound of hue (0–179) to detect cooler colours (default 160)",
    )
    parser.add_argument(
        "--sat_min",
        type=int,
        default=50,
        help="Lower bound of saturation (0–255) to filter dull colours",
    )
    parser.add_argument(
        "--val_min",
        type=int,
        default=50,
        help="Lower bound of brightness/value (0–255) to filter very dark pixels",
    )
    args = parser.parse_args()

    # Derive exclude_hot flag (default True unless --no_exclude_hot is given)
    exclude_hot = not args.no_exclude_hot

    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "void_summary.csv")

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([
            "video_id",
            "frame_number",
            "number_of_void_spots",
            "size_of_each_void_spot",
            "x_centers",
            "y_centers",
        ])

        if os.path.isfile(input_path):
            process_video(
                input_path,
                mask_dir,
                overlay_dir,
                csv_writer,
                method=args.method,
                hue_min=args.hue_min,
                hue_max=args.hue_max,
                sat_min=args.sat_min,
                val_min=args.val_min,
                min_area=args.min_area,
                max_area=args.max_area,
                kernel_size=args.kernel_size,
                circularity_threshold=args.circularity_threshold,
                use_gpu=args.use_gpu,
                sigma_small=args.sigma_small,
                sigma_large=args.sigma_large,
                exclude_hot=exclude_hot,
                hot_val_threshold=args.hot_val_threshold,
                hot_area_threshold=args.hot_area_threshold,
                otsu_ratio=args.otsu_ratio,
                multi_scale=args.multi_scale,
            )
        elif os.path.isdir(input_path):
            video_id, frame_paths = gather_png_frames(input_path)
            process_png_frames(
                frame_paths,
                video_id,
                mask_dir,
                overlay_dir,
                csv_writer,
                method=args.method,
                hue_min=args.hue_min,
                hue_max=args.hue_max,
                sat_min=args.sat_min,
                val_min=args.val_min,
                min_area=args.min_area,
                max_area=args.max_area,
                kernel_size=args.kernel_size,
                circularity_threshold=args.circularity_threshold,
                use_gpu=args.use_gpu,
                sigma_small=args.sigma_small,
                sigma_large=args.sigma_large,
                exclude_hot=exclude_hot,
                hot_val_threshold=args.hot_val_threshold,
                hot_area_threshold=args.hot_area_threshold,
                otsu_ratio=args.otsu_ratio,
                multi_scale=args.multi_scale,
            )
        else:
            raise ValueError(f"Input path {input_path} is neither a file nor a directory")

    print(f"[DONE] Processed {input_path}. Results saved to {output_dir}")


if __name__ == "__main__":
    main()