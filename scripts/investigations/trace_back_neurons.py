#!/usr/bin/env python3

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

###########################
# CONFIGURATION
###########################
DIFFERENCES_DIR = "analyses_results/differences"
LAYER_NAME = "layer_2"
MAX_FILES = 5000
SUBSAMPLE = 10000
N_COMPONENTS = 5
###########################


def run_pca_for_layer(layer_name):
    diff_files = sorted([
        f for f in os.listdir(DIFFERENCES_DIR) if f.endswith(".pt")
    ])[:MAX_FILES]
    
    all_vectors = []
    
    print(f"[INFO] Will load up to {len(diff_files)} difference files for {layer_name}...")

    for file_name in tqdm(diff_files):
        path = os.path.join(DIFFERENCES_DIR, file_name)
        data = torch.load(path)  # dict with keys: ["hidden_states", ...]

        # 1) Check if "hidden_states" is in the data
        if "hidden_states" not in data:
            continue
        
        hidden_dict = data["hidden_states"]

        # 2) Check if layer_name is in hidden_dict
        if layer_name not in hidden_dict:
            continue

        # 3) Extract the tensor and convert from bfloat16 to float32
        layer_diff = hidden_dict[layer_name].to(torch.float32)

        # 4) Move to CPU and convert to NumPy
        layer_diff_np = layer_diff.cpu().numpy()
        
        all_vectors.append(layer_diff_np)
    
    if len(all_vectors) == 0:
        print(f"[ERROR] No data found for {layer_name} in {DIFFERENCES_DIR}!")
        return
    
    # Concatenate across all examples
    all_vectors = np.concatenate(all_vectors, axis=0)
    print(f"[INFO] Combined shape before subsampling for {layer_name}: {all_vectors.shape}")

    if SUBSAMPLE < all_vectors.shape[0]:
        idx = np.random.choice(all_vectors.shape[0], size=SUBSAMPLE, replace=False)
        all_vectors = all_vectors[idx]
        print(f"[INFO] Subsampled to {all_vectors.shape[0]} tokens for {layer_name}.")

    # Run PCA
    print(f"[INFO] Running PCA on {layer_name}...")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(all_vectors)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"[RESULT] {layer_name} explained variance ratio (PC1..PC{N_COMPONENTS}): {explained_variance}")
    pc1_var_percent = explained_variance[0] * 100
    print(f"[RESULT] PC1 explains {pc1_var_percent:.4f}% of the variance for {layer_name}.")

    # Identify top neurons for PC1
    pc1_vec = pca.components_[0]  # shape [hidden_dim]
    abs_pc1 = np.abs(pc1_vec)
    top_neuron_indices = np.argsort(-abs_pc1)
    top_20 = top_neuron_indices[:20]
    
    print(f"\n[RESULT] Top 20 neurons (indices) for {layer_name} PC1:")
    for i in top_20:
        print(f"  Neuron {i} | contribution = {pc1_vec[i]:.6f}")


def main():
    # Run PCA for your default layer
    run_pca_for_layer(LAYER_NAME)

    # To run more layers in one script
    run_pca_for_layer("layer_3")

if __name__ == "__main__":
    main()
