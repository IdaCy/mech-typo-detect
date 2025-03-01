#!/usr/bin/env python3

import os
import torch
from collections import Counter
from tqdm import tqdm

###########################
# CONFIGURATION
###########################
EXTRACTIONS_DIR = "extractions"
MAX_FILES = 5000
THRESHOLD_RATIO = 10.0
MIN_TYPOS = 50
###########################


def main():
    files = sorted([
        f for f in os.listdir(EXTRACTIONS_DIR)
        if f.endswith(".pt")
    ])[:MAX_FILES]
    
    typo_token_counter = Counter()
    clean_token_counter = Counter()

    for file_name in tqdm(files):
        path = os.path.join(EXTRACTIONS_DIR, file_name)
        data = torch.load(path)

        # Structure is data["clean"]["tokens"], data["typo"]["tokens"]
        if "clean" not in data or "typo" not in data:
            continue
        
        if "tokens" not in data["clean"] or "tokens" not in data["typo"]:
            continue
        
        clean_tokens = data["clean"]["tokens"]
        typo_tokens = data["typo"]["tokens"]
        
        clean_token_counter.update(clean_tokens)
        typo_token_counter.update(typo_tokens)
    
    print("\n[INFO] Top 20 most common tokens in TYPO data:")
    for token, count in typo_token_counter.most_common(20):
        print(f"  {token}: {count}")

    print("\n[INFO] Top 20 most common tokens in CLEAN data:")
    for token, count in clean_token_counter.most_common(20):
        print(f"  {token}: {count}")

    print("\n[INFO] Checking for tokens that are frequent in TYPO vs. CLEAN:")
    suspicious_tokens = []
    for token in typo_token_counter:
        freq_typo = typo_token_counter[token]
        freq_clean = clean_token_counter.get(token, 0)
        ratio = freq_typo / (freq_clean + 1e-9)
        if ratio > THRESHOLD_RATIO and freq_typo > MIN_TYPOS:
            suspicious_tokens.append((token, freq_typo, freq_clean, ratio))
    
    suspicious_tokens.sort(key=lambda x: -x[3])
    
    print(f"\n[RESULT] Tokens that appear {THRESHOLD_RATIO}x more often in TYPO data than CLEAN (freq_typo > {MIN_TYPOS}):")
    for t, ft, fc, r in suspicious_tokens:
        print(f"  {t} | typo_freq={ft}, clean_freq={fc}, ratio={r:.2f}")


if __name__ == "__main__":
    main()
