import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Sequential, AdaptiveAvgPool2d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import pandas as pd
from dataset import FrameDataset

def get_feature_extractor(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Remove classification head, add GAP for 512-d vector
    backbone = Sequential(*list(model.children())[:-2], AdaptiveAvgPool2d((1, 1)))
    return backbone.to(device).eval()

def extract_embeddings(dataloader, model, device):
    embeddings, paths = [], []

    with torch.no_grad():
        for imgs, img_paths in tqdm(dataloader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze(-1).squeeze(-1)  # [B, 512, 1, 1] → [B, 512]
            embeddings.append(feats.cpu().numpy())
            paths.extend(img_paths)

    return np.vstack(embeddings), paths

def cluster_embeddings(embeddings, n_clusters=10):
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reduced)
    iso = IsolationForest(contamination=0.01, random_state=42).fit(reduced)

    return {
        "reduced": reduced,
        "kmeans_labels": kmeans.labels_,
        "isolation_scores": iso.decision_function(reduced),
        "is_outlier": iso.predict(reduced)  # -1 = outlier
    }

def to_rgb_channels(x):
    return x.repeat(3, 1, 1)

def save_summary(paths, results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.DataFrame({
        "path": paths,
        "kmeans_label": results["kmeans_labels"],
        "isolation_score": results["isolation_scores"],
        "is_outlier": results["is_outlier"]
    })
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved summary: {output_file}")

def main(image_dir, output_csv, batch_size=32, n_clusters=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transform for pretrained model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        to_rgb_channels,
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = FrameDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f"[INFO] Loaded {len(dataset)} images from {image_dir}")
    model = get_feature_extractor(device)
    embeddings, paths = extract_embeddings(dataloader, model, device)
    results = cluster_embeddings(embeddings, n_clusters=n_clusters)
    save_summary(paths, results, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to PNG image frames")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV with clustering info")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_clusters", type=int, default=10)
    args = parser.parse_args()

    main(args.image_dir, args.output_csv, args.batch_size, args.n_clusters)
print(f"Cuda available: {torch.cuda.is_available()}")