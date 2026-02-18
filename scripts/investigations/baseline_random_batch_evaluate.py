#!/usr/bin/env python3

"""
evaluate_random_patch_baseline.py

Reads random_patch_baseline.csv produced by randomized_patching_baseline.py.
Calculates average perplexities for each approach (no patch, PC1 patch, random patch)
and prints out the difference or ratio for quick comparison.
"""

import csv
import sys
import math
import statistics

INPUT_CSV = "analyses_results/random_patch_baseline.csv"

def main():
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("ERROR: CSV is empty or has no header.")
            sys.exit(1)

        # Expecting columns: original_prompt, text_none, ppl_none, text_pc1, ppl_pc1, text_rand, ppl_rand
        # We'll find these column indices to be robust
        try:
            idx_ppl_none = header.index("ppl_none")
            idx_ppl_pc1  = header.index("ppl_pc1")
            idx_ppl_rand = header.index("ppl_rand")
        except ValueError as e:
            print(f"ERROR: Missing expected columns in CSV: {e}")
            sys.exit(1)

        for row in reader:
            if not row:
                continue
            try:
                ppl_none = float(row[idx_ppl_none])
                ppl_pc1  = float(row[idx_ppl_pc1])
                ppl_rand = float(row[idx_ppl_rand])
                rows.append((ppl_none, ppl_pc1, ppl_rand))
            except ValueError:
                # Maybe the row is incomplete or has invalid numbers
                continue

    if not rows:
        print(f"No valid data rows found in {INPUT_CSV}. Exiting.")
        sys.exit(0)

    # Gather perplexities in separate lists
    none_list  = [r[0] for r in rows]
    pc1_list   = [r[1] for r in rows]
    rand_list  = [r[2] for r in rows]

    # Compute stats
    avg_none   = statistics.mean(none_list)
    avg_pc1    = statistics.mean(pc1_list)
    avg_rand   = statistics.mean(rand_list)

    std_none   = statistics.pstdev(none_list)  # population stdev, or use statistics.stdev
    std_pc1    = statistics.pstdev(pc1_list)
    std_rand   = statistics.pstdev(rand_list)

    print(f"Loaded {len(rows)} data rows from {INPUT_CSV}")
    print("=== Average Perplexities ===")
    print(f"  no patch:   {avg_none:.3f} ± {std_none:.3f}")
    print(f"  PC1 patch:  {avg_pc1:.3f} ± {std_pc1:.3f}")
    print(f"  random:     {avg_rand:.3f} ± {std_rand:.3f}")

    # Maybe compare them:
    diff_pc1_none  = avg_pc1 - avg_none
    diff_rand_none = avg_rand - avg_none

    print("\n=== Differences in Perplexity vs. No Patch ===")
    print(f"  PC1 patch - no patch = {diff_pc1_none:.3f}")
    print(f"  random   - no patch = {diff_rand_none:.3f}")

    # Possibly a ratio:
    ratio_pc1_none  = avg_pc1 / avg_none if avg_none != 0 else float('inf')
    ratio_rand_none = avg_rand / avg_none if avg_none != 0 else float('inf')
    print("\n=== Ratios in Perplexity vs. No Patch ===")
    print(f"  PC1 patch / no patch = {ratio_pc1_none:.3f}")
    print(f"  random   / no patch = {ratio_rand_none:.3f}")

if __name__ == "__main__":
    main()

