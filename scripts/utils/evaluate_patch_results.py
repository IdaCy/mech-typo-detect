#!/usr/bin/env python3

"""
consolidate_all_results.py

A meta-script to gather and analyze the results from:

1) attention_analysis.py => produces "analyses_results/attention_analysis.csv"
2) logit_analysis.py     => produces "analyses_results/logit_analysis.csv"
3) run_dimension_causal_experiment_single.py => produces "dimension_causal_S_{typo/clean}.csv"

We read each CSV, parse or compute interesting summary stats, and write out a combined summary.

"""

import os
import csv
import numpy as np

# =====================
# CONFIG
# =====================
ATTENTION_CSV = "analyses_results/attention_analysis.csv"
LOGIT_CSV     = "analyses_results/logit_analysis.csv"
CAUSAL_CSV    = "analyses_results/dimension_causal_S_typo.csv"  # or dimension_causal_S_clean.csv
OUTPUT_CSV    = "analyses_results/consolidated_summary.csv"

def read_attention_csv(path):
    """
    Expected columns from attention_analysis.py script:
    layer, count_samples, mean_diff, std_diff
    We'll parse them into a dict: attn_summary[layer] = (count_samples, mean_diff, std_diff)
    """
    attn_summary = {}
    if not os.path.exists(path):
        print(f"[WARN] No attention CSV found at {path} - skipping.")
        return attn_summary

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return attn_summary

        # Expect: [layer, count_samples, mean_diff, std_diff]
        for row in reader:
            if len(row) < 4:
                continue
            layer_str, count_str, mean_str, std_str = row
            try:
                layer = int(layer_str)
                count_samples = int(count_str)
                mean_diff = float(mean_str)
                std_diff  = float(std_str)
                attn_summary[layer] = {
                    "count": count_samples,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff
                }
            except ValueError:
                continue
    return attn_summary

def read_logit_csv(path):
    """
    From logit_analysis.py (Revised):
    Fields: file, num_tokens, avg_kl, frac_top1_same
    We'll store them in a list of dicts for now.
    """
    if not os.path.exists(path):
        print(f"[WARN] No logit CSV found at {path} - skipping.")
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

def read_causal_csv(path):
    """
    From run_dimension_causal_experiment_single.py
    We have rows:
      prompt, text_normal, ppl_normal, text_removeDim, ppl_removeDim, (optionally text_addDim, ppl_addDim)
    We'll store them for summary stats.
    """
    if not os.path.exists(path):
        print(f"[WARN] No causal CSV found at {path} - skipping.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    # The first row is the header, we can parse or skip
    if not all_rows:
        return []

    header = all_rows[0]
    data_rows = all_rows[1:]

    # We'll check if we have addDim or not
    # e.g. header = ["prompt", "text_normal", "ppl_normal", "text_removeDim", "ppl_removeDim", ...]
    # We'll assume a fixed structure or adapt
    has_add = (len(header) == 8)

    # We'll gather perplexities
    out_list = []
    for r in data_rows:
        if has_add:
            if len(r) < 8:
                continue
            prompt, text_norm, ppl_norm, text_rm, ppl_rm, text_add, ppl_add = r
        else:
            if len(r) < 5:
                continue
            prompt, text_norm, ppl_norm, text_rm, ppl_rm = r
            text_add, ppl_add = None, None

        # parse floats for perplexities
        try:
            ppl_n = float(ppl_norm)
            ppl_r = float(ppl_rm)
        except ValueError:
            ppl_n, ppl_r = None, None

        if text_add is not None and ppl_add is not None:
            try:
                ppl_a = float(ppl_add)
            except ValueError:
                ppl_a = None
        else:
            ppl_a = None

        out_list.append({
            "prompt": prompt,
            "ppl_normal": ppl_n,
            "ppl_remove": ppl_r,
            "ppl_add": ppl_a
        })
    return out_list


def main():
    # 1) Read each CSV
    attn_dict = read_attention_csv(ATTENTION_CSV)   # -> dict
    logit_list= read_logit_csv(LOGIT_CSV)           # -> list of dict
    causal_list= read_causal_csv(CAUSAL_CSV)        # -> list of dict

    # 2) Summarize each
    # =============== ATTENTION
    # attn_dict has layer -> {count, mean_diff, std_diff}
    # We'll just print them out or store in final CSV
    # =============== LOGITS
    # logit_list has each row: { file, num_tokens, avg_kl, frac_top1_same }
    # We can do an average if we want
    if logit_list:
        kl_values = []
        top1_values = []
        for row in logit_list:
            try:
                klv = float(row["avg_kl"])
                t1 = float(row["frac_top1_same"])
                kl_values.append(klv)
                top1_values.append(t1)
            except ValueError:
                continue
        mean_kl = np.mean(kl_values) if kl_values else 0.0
        std_kl  = np.std(kl_values)  if kl_values else 0.0
        mean_t1 = np.mean(top1_values) if top1_values else 0.0
        std_t1  = np.std(top1_values)  if top1_values else 0.0
    else:
        mean_kl, std_kl = 0.0, 0.0
        mean_t1, std_t1 = 0.0, 0.0

    # =============== CAUSAL
    # We can compute average perplexity difference, etc.
    if causal_list:
        diffs = []
        # if there's ppl_add, we might also store that
        for row in causal_list:
            if row["ppl_normal"] is None or row["ppl_remove"] is None:
                continue
            diff = row["ppl_remove"] - row["ppl_normal"]
            diffs.append(diff)
        if diffs:
            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))
        else:
            mean_diff, median_diff = 0.0, 0.0
    else:
        mean_diff, median_diff = 0.0, 0.0

    # 3) Write final summary
    summary_path = OUTPUT_CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Analysis", "Value"])
        writer.writerow(["--- ATTENTION ANALYSIS ---",""])
        for layer, vals in sorted(attn_dict.items()):
            c = vals["count"]
            md= vals["mean_diff"]
            sd= vals["std_diff"]
            writer.writerow([f"Layer {layer} (count={c}) mean_diff", f"{md:.6f} ± {sd:.6f}"])

        writer.writerow(["--- LOGIT ANALYSIS ---",""])
        writer.writerow(["Mean KL (clean vs. typo)", f"{mean_kl:.6f} ± {std_kl:.6f}"])
        writer.writerow(["Mean fraction top1 same", f"{mean_t1:.4f} ± {std_t1:.4f}"])

        writer.writerow(["--- DIMENSION CAUSAL EXPERIMENT ---",""])
        writer.writerow(["Mean perplexity difference (remove - normal)", f"{mean_diff:.4f}"])
        writer.writerow(["Median perplexity difference (remove - normal)", f"{median_diff:.4f}"])

    print(f"[INFO] Wrote consolidated summary to {summary_path}")
    print("[INFO] Done. Check the CSV for aggregated results.")


if __name__ == "__main__":
    main()
