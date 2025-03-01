import os
import torch
import matplotlib.pyplot as plt

# Allow the global function that PyTorch is complaining about.
torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])

# Define paths for the PCA results.
pca_dir = "analyses_results/PCA_PC1"
results_file = os.path.join(pca_dir, "layer_pca_results.pt")
pc1_file = os.path.join(pca_dir, "layer_pc1_vectors.pt")

# Check that the files exist.
if not os.path.exists(results_file):
    raise ValueError(f"PCA results file not found: {results_file}")
if not os.path.exists(pc1_file):
    raise ValueError(f"PC1 vectors file not found: {pc1_file}")

# Load the PCA results.
pca_results = torch.load(results_file, map_location="cpu", weights_only=False)
pc1_vectors = torch.load(pc1_file, map_location="cpu", weights_only=False)

print("=== PCA Results Summary ===")
print("Type of PCA results:", type(pca_results))
print("Layers found:", sorted(pca_results.keys(), key=lambda x: int(x.split('_')[1])))
print(f"Total layers: {len(pca_results)}\n")

for layer in sorted(pca_results.keys(), key=lambda x: int(x.split('_')[1])):
    ev = pca_results[layer]
    print(f"Layer {layer}:")
    print(f"  Explained variance ratios (top 10): {ev}")
    print("")

print("=== PC1 Vectors Summary ===")
print("Type of PC1 vectors:", type(pc1_vectors))
print("Layers found:", sorted(pc1_vectors.keys(), key=lambda x: int(x.split('_')[1])))
print(f"Total layers: {len(pc1_vectors)}\n")

for layer in sorted(pc1_vectors.keys(), key=lambda x: int(x.split('_')[1])):
    vec = pc1_vectors[layer]
    print(f"Layer {layer}:")
    print(f"  PC1 vector shape: {vec.shape}")
    # Print first 5 elements for a quick look.
    print(f"  First 5 elements: {vec[:5]}")
    print("")

# Plot the explained variance ratio of PC1 across layers.
# Plot the explained variance ratio of PC1 across layers.
layers = sorted(pca_results.keys(), key=lambda x: int(x.split("_")[1]))
numeric_layers = [int(k.split("_")[1]) for k in layers]
first_pc_ev = [pca_results[layer][0] for layer in layers]

plt.figure(figsize=(10, 5))
plt.plot(numeric_layers, first_pc_ev, marker='o', linestyle='-')
plt.xlabel("Layer")
plt.ylabel("Explained Variance Ratio (PC1)")
plt.title("PC1 Explained Variance Ratio for Relevant Layers")
plt.grid(True)

# Limit y-axis if desired
plt.ylim(0.95, 1.0)

plt.tight_layout()
plt.show()
