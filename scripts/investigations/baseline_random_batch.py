#!/usr/bin/env python3

"""
Demonstrates a patching experiment with PC1 vs. a random direction of the same dimensionality.

Steps:
1) Load a model & tokenizer (e.g. Mistral).
2) Load your PC1 vector from a .pt file (or define it inline).
3) Generate a random vector of the same size & norm.
4) Provide hooks that patch the hidden states with either PC1 or this random vector.
5) Compare perplexities & final text of each approach (no patch vs PC1 patch vs random patch).
"""

import os
import csv
import math
import torch
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# SCOPE/SPEED CONFIG
DATASET_CSV       = "prompts/preprocessed/typoQs.csv"   # Each row is a prompt
SUBSAMPLE_LINES   = 20                                  # Limit lines for demonstration
MODEL_NAME        = "../typo-correct-subspaces/models/mistral-7b"
DEVICE            = "cuda"

PC1_FILE          = "analyses_results/PCA_PC1/layer_pc1_vectors.pt"
PC1_KEY           = "layer_2"

OUTPUT_CSV        = "analyses_results/random_patch_baseline.csv"
LOG_FILE          = "logs/random_patch_baseline.log"

LAYER_INDEX       = 2
SCALE             = 1.0 # scaling factor for patch

MAX_NEW_TOKENS    = 40
DO_SAMPLE         = True
TEMPERATURE       = 0.7
TOP_P             = 0.9

# For e.g. "Yes or no: does this sentence contain a typo"
PROMPT_PREFIX     = ""

# -----------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    loss_val = out.loss.item()
    return math.exp(loss_val)

def main():
    logger.info("=== Starting randomized_patching_baseline.py ===")

    # 1) Load model & tokenizer
    logger.info(f"Loading model {MODEL_NAME} to device={DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    # 2) Load PC1
    if not os.path.exists(PC1_FILE):
        logger.error(f"PC1 file not found: {PC1_FILE}")
        return
    pc1_dict = torch.load(PC1_FILE, map_location="cpu")
    if PC1_KEY not in pc1_dict:
        logger.error(f"PC1 key='{PC1_KEY}' not found in {list(pc1_dict.keys())}")
        return
    pc1_vector = pc1_dict[PC1_KEY]
    if not isinstance(pc1_vector, torch.Tensor):
        pc1_vector = torch.tensor(pc1_vector)
    pc1_vector = pc1_vector.to(torch.float32).to(DEVICE)
    logger.info(f"Loaded PC1 vector {PC1_KEY}, shape={pc1_vector.shape}")
    # e.g. shape [hidden_dim]

    # 3) Create a random vector of same shape & norm
    random_vector = torch.randn_like(pc1_vector, dtype=torch.float32, device=DEVICE)
    random_norm   = random_vector.norm()
    if random_norm > 0:
        random_vector /= random_norm
    logger.info("Generated random vector with same dimension as PC1, normalized to 1.")

    # Patch function for PC1: we do a simple "output_ += scale * outer(...)"
    def patch_pc1(module, input_, output_):
        output_ += SCALE * pc1_vector.unsqueeze(0).unsqueeze(0)
        return output_

    # Patch function for random vector
    def patch_random(module, input_, output_):
        output_ += SCALE * random_vector.unsqueeze(0).unsqueeze(0)
        return output_

    # We’ll define a helper to register one of these patch functions on the layer
    layer_module = model.model.layers[LAYER_INDEX].mlp
    def register_hook(patch_fn):
        return layer_module.register_forward_hook(patch_fn)

    # 4) Read input lines
    if not os.path.exists(DATASET_CSV):
        logger.error(f"Dataset CSV not found: {DATASET_CSV}")
        return
    lines = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                lines.append(row[0].strip())

    if SUBSAMPLE_LINES and len(lines) > SUBSAMPLE_LINES:
        lines = lines[:SUBSAMPLE_LINES]
    logger.info(f"Loaded {len(lines)} lines from {DATASET_CSV}")

    # 5) Generation function
    def generate_text(prompt: str, mode: str) -> str:
        """
        mode = 'none', 'pc1', or 'random'.
        We'll register the appropriate hook if needed, generate, remove hook.
        """
        handle = None
        if mode == 'pc1':
            handle = register_hook(patch_pc1)
        elif mode == 'random':
            handle = register_hook(patch_random)

        enc = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
        text_out = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if handle is not None:
            handle.remove()

        return text_out

    # 6) Main loop: do normal vs pc1 vs random for each line
    out_header = [
        "original_prompt",
        "text_none", "ppl_none",
        "text_pc1",  "ppl_pc1",
        "text_rand", "ppl_rand"
    ]
    out_rows = [out_header]
    logger.info("Beginning generation with no patch, pc1 patch, and random patch...")

    for i, line in enumerate(tqdm(lines, desc="RandomPatching")):
        prompt = PROMPT_PREFIX + line

        # no patch
        text_none = generate_text(prompt, mode='none')
        ppl_none  = compute_perplexity(model, tokenizer, text_none)

        # pc1 patch
        text_pc1 = generate_text(prompt, mode='pc1')
        ppl_pc1  = compute_perplexity(model, tokenizer, text_pc1)

        # random patch
        text_rand = generate_text(prompt, mode='random')
        ppl_rand  = compute_perplexity(model, tokenizer, text_rand)

        row = [
            line,
            text_none, f"{ppl_none:.3f}",
            text_pc1,  f"{ppl_pc1:.3f}",
            text_rand, f"{ppl_rand:.3f}"
        ]
        out_rows.append(row)

    # 7) Save results
    logger.info(f"Writing results to {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    logger.info("=== Completed randomized_patching_baseline.py ===")

if __name__ == "__main__":
    main()

