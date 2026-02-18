#!/usr/bin/env python3

"""
compare_typos.py

Generates multiple variants of each prompt with different types of typos:
1) Missing a letter
2) Swapped adjacent letters
3) Repeated a letter
Then runs inference or logs the same neuron activation to compare.
"""

import random
import os

def introduce_typo_missing(word):
    # remove one random character, ignoring very short words
    if len(word) > 2:
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx+1:]
    return word

def introduce_typo_swapped(word):
    # swap two adjacent letters
    if len(word) > 2:
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    return word

def introduce_typo_repeated(word):
    if len(word) > 2:
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx] + word[idx] + word[idx+1:]
    return word

def main():
    CLEAN_FILE = "prompts/preprocessed/cleanQs.csv"
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # We'll produce 3 new files
    out_missing = open("typo_missing.csv", "w", encoding="utf-8")
    out_swapped = open("typo_swapped.csv", "w", encoding="utf-8")
    out_repeated= open("typo_repeated.csv", "w", encoding="utf-8")

    for line in lines:
        words = line.split()
        # pick a random word that's not too short
        valid_words = [w for w in words if len(w) > 3]
        if not valid_words:
            # just copy line if we can't find a suitable word
            out_missing.write(line + "\n")
            out_swapped.write(line + "\n")
            out_repeated.write(line + "\n")
            continue

        target_word = random.choice(valid_words)
        # produce 3 variants
        missing = " ".join(w if w != target_word else introduce_typo_missing(w) for w in words)
        swapped = " ".join(w if w != target_word else introduce_typo_swapped(w) for w in words)
        repeated= " ".join(w if w != target_word else introduce_typo_repeated(w) for w in words)

        out_missing.write(missing + "\n")
        out_swapped.write(swapped + "\n")
        out_repeated.write(repeated+ "\n")

    out_missing.close()
    out_swapped.close()
    out_repeated.close()

    print("Generated typo_missing.csv, typo_swapped.csv, typo_repeated.csv.")

if __name__ == "__main__":
    main()
