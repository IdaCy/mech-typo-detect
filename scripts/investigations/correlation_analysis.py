import torch
import numpy as np
import os

# Load a sample file
file_path = "analyses_results/differences/activations_00000.pt"
data = torch.load(file_path, map_location="cpu")

# Select a layer to inspect
layer = "layer_2"  # Change as needed
if "hidden_states" in data and layer in data["hidden_states"]:
    tensor = data["hidden_states"][layer]  # Shape: [tokens, 4096]
    array = tensor.numpy()
    
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(array, rowvar=False)

    print("Correlation matrix shape:", correlation_matrix.shape)
    print("Mean correlation (excluding diagonal):", np.mean(correlation_matrix - np.eye(array.shape[1])))

    # Check if the first eigenvalue dominates
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    print("Top 5 eigenvalues:", eigenvalues[-5:])  # Largest eigenvalues
