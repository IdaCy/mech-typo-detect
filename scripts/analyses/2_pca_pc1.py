import os
import torch
import glob
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directories for input differences and output analyses.
# IMPORTANT: Update this path to where your differences files are stored.
# For example, if your files are saved under "analyses_results/tests", change diff_dir accordingly.
diff_dir = "analyses_results/differences"  
output_dir = "analyses_results/PCA"
os.makedirs(output_dir, exist_ok=True)

# List all difference files (they are in .pt format).
diff_files = sorted(glob.glob(os.path.join(diff_dir, "*.pt")))
print("Total difference files found:", len(diff_files))

if len(diff_files) == 0:
    raise ValueError("No difference files found in directory: " + diff_dir)

# Determine the number of layers by loading one file and checking the keys in hidden_states.
sample_data = torch.load(diff_files[0], map_location="cpu")
if isinstance(sample_data, dict) and "hidden_states" in sample_data:
    # Get all keys starting with "layer_"
    layer_keys = [key for key in sample_data["hidden_states"].keys() if key.startswith("layer_")]
    # Sort keys by their numeric part.
    layer_keys = sorted(layer_keys, key=lambda k: int(k.split("_")[1]))
    num_layers = len(layer_keys)
else:
    raise ValueError("Unexpected format in difference file. Expected dict with 'hidden_states'.")
print("Number of layers detected:", num_layers)

# We'll store PCA results (explained variance ratios) and PC1 vectors for each layer.
layer_pca_results = {}
layer_pc1_vectors = {}

# Set maximum number of vectors per layer to use in PCA (to manage memory/computation).
max_samples = 10000
# Use 8 worker threads (matching your 8 cores)
num_workers = 8

def process_file_for_layer(file, layer):
    """
    Loads one difference file and returns the flattened difference vectors
    for the specified layer. Uses key "layer_{layer}".
    """
    try:
        diff_data = torch.load(file, map_location="cpu")
        if isinstance(diff_data, dict) and "hidden_states" in diff_data:
            hidden_states = diff_data["hidden_states"]
            key = f"layer_{layer}"
            if key in hidden_states:
                tensor = hidden_states[key]  # Expected shape: [num_tokens, hidden_dim]
                flat = tensor.reshape(-1, tensor.shape[-1])
                return flat.numpy()
    except Exception as e:
        print(f"Error processing {file} for layer {layer}: {e}")
    return None

for layer in range(num_layers):
    print(f"\nProcessing layer {layer}...")
    all_diff_vectors_list = []
    # Process files in parallel using ThreadPoolExecutor.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_for_layer, file, layer): file for file in diff_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Layer {layer} files"):
            result = future.result()
            if result is not None:
                all_diff_vectors_list.append(result)
    if len(all_diff_vectors_list) == 0:
        print(f"No data collected for layer {layer}.")
        continue
    # Concatenate all arrays from this layer.
    all_diff_vectors = np.concatenate(all_diff_vectors_list, axis=0)
    print(f"Collected {all_diff_vectors.shape[0]} vectors for layer {layer}.")
    
    # Subsample if necessary.
    if all_diff_vectors.shape[0] > max_samples:
        indices = np.random.choice(all_diff_vectors.shape[0], size=max_samples, replace=False)
        all_diff_vectors = all_diff_vectors[indices]
        print(f"Subsampled to {max_samples} vectors for layer {layer}.")
    
    # Run PCA on the difference vectors for this layer.
    pca = PCA(n_components=10)
    pca.fit(all_diff_vectors)
    explained_variance = pca.explained_variance_ratio_
    layer_pca_results[layer] = explained_variance
    # Save the first principal component (PC1).
    pc1 = pca.components_[0]  # Shape: [hidden_dim]
    layer_pc1_vectors[layer] = pc1
    print(f"Layer {layer}: Top 10 explained variance ratios: {explained_variance}")

# Save the PCA results and PC1 vectors to the output directory.
results_file = os.path.join(output_dir, "layer_pca_results.pt")
torch.save(layer_pca_results, results_file)
print(f"PCA results saved to {results_file}")

pc1_file = os.path.join(output_dir, "layer_pc1_vectors.pt")
torch.save(layer_pc1_vectors, pc1_file)
print(f"PC1 vectors saved to {pc1_file}")

# Optionally, plot the PC1 explained variance (the variance of the first principal component) across layers.
layers = sorted(layer_pca_results.keys())
first_pc = [layer_pca_results[layer][0] for layer in layers]

plt.figure(figsize=(10, 5))
plt.plot(layers, first_pc, marker='o')
plt.xlabel("Layer")
plt.ylabel("Explained Variance Ratio (PC1)")
plt.title("PC1 Explained Variance Ratio per Layer")
plt.grid(True)
plot_file = os.path.join(output_dir, "pca_plot.png")
plt.savefig(plot_file)
plt.close()
print(f"PCA plot saved to {plot_file}")
