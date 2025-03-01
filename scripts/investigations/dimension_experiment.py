#!/usr/bin/env python3

"""
run_dimension_causal_experiment_single.py

A script to run either on a "typo" dataset or a "clean" dataset, depending on a toggle.
It:
1) Loads Mistral model & tokenizer
2) Loads PC1 vector from your existing "layer_pc1_vectors.pt" (the PCA output).
3) Reads a single CSV of prompts (either typed or clean).
4) For each prompt, generates:
   - Normal text (no hooking).
   - Dimension-removed text (subtract projection on PC1).
   - (Optional) dimension-added text if you want meltdown check.
5) Writes final texts & perplexities to a CSV.

You can switch between a "typo" run or a "clean" run by changing the 'DATASET_CHOICE' or 'INPUT_FILE' variables.
"""

import os
import csv
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "../typo-correct-subspaces/models/mistral-7b"
PC1_VECTORS_PATH = "analyses_results/PCA_PC1/layer_pc1_vectors.pt"  # from your PCA script
PC1_LAYER_KEY = "layer_2"  # which layer's PC1 to use
LAYER_INDEX = 2
DEVICE = "cuda"

# Decide which dataset to run. E.g. "typo" or "clean".
DATASET_CHOICE = "typo"  # or "clean"

# If you want separate files or toggles:
INPUT_FILE_TYPO  = "prompts/preprocessed/cleanQs.csv"
INPUT_FILE_CLEAN = "prompts/preprocessed/typoQs.csv"

# We'll output a single CSV with hooking results
OUTPUT_CSV_FILE  = "analyses_results/dimension_causal_" + DATASET_CHOICE + ".csv"

# Generation config
MAX_NEW_TOKENS   = 80
DO_SAMPLE        = True
TEMPERATURE      = 0.7
TOP_P            = 0.9

# If True, do "dimension added" hooking as well.
PERFORM_ADD_TEST = True

# -----------------------
def compute_perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())

def main():
    # 1) Pick the input file based on DATASET_CHOICE
    if DATASET_CHOICE == "typo":
        INPUT_FILE = INPUT_FILE_TYPO
    elif DATASET_CHOICE == "clean":
        INPUT_FILE = INPUT_FILE_CLEAN
    else:
        raise ValueError(f"Unknown DATASET_CHOICE: {DATASET_CHOICE}")

    print(f"[INFO] Using dataset = {DATASET_CHOICE}, from file = {INPUT_FILE}")

    # 2) Load model & tokenizer
    print("[INFO] Loading model & tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    # 3) Load PC1 vector from the PCA dictionary
    if not os.path.exists(PC1_VECTORS_PATH):
        raise FileNotFoundError(f"{PC1_VECTORS_PATH} not found!")
    pca_data = torch.load(PC1_VECTORS_PATH, map_location="cpu")
    if PC1_LAYER_KEY not in pca_data:
        raise KeyError(f"'{PC1_LAYER_KEY}' not in keys: {list(pca_data.keys())}")

    pc1_raw = pca_data[PC1_LAYER_KEY]
    if isinstance(pc1_raw, np.ndarray):
        pc1_vector = torch.from_numpy(pc1_raw).float()
    elif torch.is_tensor(pc1_raw):
        pc1_vector = pc1_raw.float()
    else:
        raise ValueError(f"Unexpected type for PC1 vector in {PC1_LAYER_KEY}")

    pc1_vector = pc1_vector.to(DEVICE)
    hidden_dim = pc1_vector.shape[0]
    print(f"[INFO] Loaded PC1 vector for {PC1_LAYER_KEY}, shape={pc1_vector.shape} (layer={LAYER_INDEX}).")

    # 4) Define hooking
    pc1_norm_sq = torch.sum(pc1_vector * pc1_vector)

    def remove_dim_hook_fn(module, input_, output_):
        pc1_ = pc1_vector.unsqueeze(0).unsqueeze(0)
        dot = (output_ * pc1_).sum(dim=-1, keepdim=True)
        proj = dot / pc1_norm_sq * pc1_
        return output_ - proj

    def add_dim_hook_fn(module, input_, output_):
        pc1_ = pc1_vector.unsqueeze(0).unsqueeze(0)
        return output_ + pc1_

    # 5) Access layer submodule
    layer_module = model.model.layers[LAYER_INDEX].mlp

    def generate_text(prompt: str, hook=None):
        if hook is not None:
            handle = layer_module.register_forward_hook(hook)
        else:
            handle = None

        enc = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if handle is not None:
            handle.remove()
        return text

    # 6) Read prompts
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found!")
    lines = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                lines.append(row[0].strip())

    print(f"[INFO] Found {len(lines)} prompts in {INPUT_FILE}.")

    # 7) For each prompt, do normal vs. dimension-removed (and dimension-added if desired)
    out_header = [
        "prompt",
        "text_normal", "ppl_normal",
        "text_removeDim", "ppl_removeDim"
    ]
    if PERFORM_ADD_TEST:
        out_header += ["text_addDim", "ppl_addDim"]

    out_rows = [out_header]

    for prompt in tqdm(lines, desc=f"{DATASET_CHOICE.capitalize()}Exp"):
        text_normal = generate_text(prompt, hook=None)
        ppl_normal  = compute_perplexity(model, tokenizer, text_normal)

        text_remove = generate_text(prompt, hook=remove_dim_hook_fn)
        ppl_remove  = compute_perplexity(model, tokenizer, text_remove)

        row = [
            prompt,
            text_normal, f"{ppl_normal:.3f}",
            text_remove, f"{ppl_remove:.3f}"
        ]

        if PERFORM_ADD_TEST:
            text_add = generate_text(prompt, hook=add_dim_hook_fn)
            ppl_add  = compute_perplexity(model, tokenizer, text_add)
            row += [text_add, f"{ppl_add:.3f}"]

        out_rows.append(row)

    # 8) Write output
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    with open(OUTPUT_CSV_FILE, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    print(f"[INFO] Done. Results saved to {OUTPUT_CSV_FILE}.")
    print("Preview last row:", out_rows[-1])

if __name__ == "__main__":
    main()

