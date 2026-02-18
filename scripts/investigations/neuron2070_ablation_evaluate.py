#!/usr/bin/env python3

"""
analyze_neuron_ablation.py

Reads the CSV produced by 'neuron2070_ablation.py', which has columns:

    prompt, text_normal, ppl_normal, text_neuronAblated, ppl_neuronAblated

We compute:
1) Average normal perplexity and average ablated perplexity
2) Distribution of (ablated_ppl - normal_ppl)
3) Fraction of prompts where ablation improved perplexity (lower is better)
4) Fraction of prompts where final text changed
5) A histogram of perplexity differences

Requires:
    pip install matplotlib numpy
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Path to the CSV generated with 'neuron2070_ablation.py'
INPUT_CSV = "analyses_results/neuron2070_ablation.csv"
OUTPUT_DIR = "analyses_results/neuron2070_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Could not find input CSV: {INPUT_CSV}")

    prompts = []
    text_normal_list = []
    ppl_normal_list = []
    text_ablate_list = []
    ppl_ablate_list = []

    # Read the CSV
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip the header row
        if not header or len(header) < 5:
            raise ValueError("Unexpected CSV format. Expect at least 5 columns.")
        for row in reader:
            # row = [prompt, text_normal, ppl_normal, text_neuronAblated, ppl_neuronAblated]
            if len(row) < 5:
                continue
            prompt          = row[0]
            text_normal     = row[1]
            ppl_normal_str  = row[2]
            text_ablate     = row[3]
            ppl_ablate_str  = row[4]

            # Convert perplexities to float
            try:
                ppl_normal_val = float(ppl_normal_str)
                ppl_ablate_val = float(ppl_ablate_str)
            except:
                # skip malformed lines
                continue

            prompts.append(prompt)
            text_normal_list.append(text_normal)
            ppl_normal_list.append(ppl_normal_val)
            text_ablate_list.append(text_ablate)
            ppl_ablate_list.append(ppl_ablate_val)

    # Convert to np arrays
    ppl_normal_arr = np.array(ppl_normal_list)
    ppl_ablate_arr = np.array(ppl_ablate_list)

    # 1) Basic stats
    mean_normal = np.mean(ppl_normal_arr)
    mean_ablate = np.mean(ppl_ablate_arr)
    median_normal = np.median(ppl_normal_arr)
    median_ablate = np.median(ppl_ablate_arr)

    print("=== Perplexity Stats ===")
    print(f"Number of samples read: {len(ppl_normal_arr)}")
    print(f"Mean normal perplexity:  {mean_normal:.3f}")
    print(f"Mean ablated perplexity: {mean_ablate:.3f}")
    print(f"Median normal perplexity:  {median_normal:.3f}")
    print(f"Median ablated perplexity: {median_ablate:.3f}")

    # 2) Distribution of (ppl_ablate - ppl_normal)
    ppl_diff = ppl_ablate_arr - ppl_normal_arr
    mean_diff = np.mean(ppl_diff)
    median_diff = np.median(ppl_diff)
    print("\n=== Perplexity Difference: ablated - normal ===")
    print(f"Mean difference:   {mean_diff:.3f}")
    print(f"Median difference: {median_diff:.3f}")

    # 3) How often the text changed
    changes = 0
    for tnorm, tabl in zip(text_normal_list, text_ablate_list):
        if tnorm.strip() != tabl.strip():
            changes += 1
    frac_changed_text = changes / len(text_normal_list)
    print(f"Fraction of final text changed by ablation: {frac_changed_text:.2%}")

    # 5) Plot histogram of (ppl_ablate - ppl_normal)
    plt.figure(figsize=(6,4))
    plt.hist(ppl_diff, bins=40, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='No difference')
    plt.xlabel("Ablated perplexity - Normal perplexity")
    plt.ylabel("Count")
    plt.title("Distribution of Perplexity Differences (Ablated - Normal)")
    plt.grid(True)
    plt.legend()
    out_plot = os.path.join(OUTPUT_DIR, "perplexity_diff_hist.png")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    print(f"\nSaved perplexity difference histogram to {out_plot}")

    print("\n=== Done with analysis of neuron ablation results. ===")

if __name__ == "__main__":
    main()
