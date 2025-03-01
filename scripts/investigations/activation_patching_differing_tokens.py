#!/usr/bin/env python3

"""
Demonstrates token-level hooking removing the typo dimension
at the tokens that differ from clean -> typo. We do:

1) Read a CSV with columns [clean, typo].
2) For each row, tokenize both strings, use difflib to find mismatch token indices.
3) Run normal generation for the typo text, measure perplexity.
4) Run token-level hooking (only removing PC1 at mismatch tokens), measure perplexity.
5) Write final text & perplexities to a CSV.

Requires to already have:
- A PCA file with "layer_pc1_vectors.pt" storing e.g. pc1 for "layer_2"
- A parallel CSV: [clean_text, typo_text]
"""

import os
import csv
import math
import torch
import logging
import numpy as np
import difflib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------
# GLOBAL CONFIG
# -----------------------
PARALLEL_CSV       = "prompts/preprocessed/typoQs.csv"  # 2 columns: [clean, typo]
SUBSAMPLE_LINES    = 5000
MODEL_NAME         = "../typo-correct-subspaces/models/mistral-7b"
PC1_VECTORS_PATH   = "analyses_results/PCA_PC1/layer_pc1_vectors.pt"
PC1_LAYER_KEY      = "layer_2"
LAYER_INDEX        = 2
DEVICE             = "cuda"

OUTPUT_CSV         = "analyses_results/token_level_hook_difflib_results.csv"
LOG_FILE           = "logs/token_level_hook_difflib.log"

MAX_NEW_TOKENS     = 40
DO_SAMPLE          = True
TEMPERATURE        = 0.7
TOP_P              = 0.9
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
    return math.exp(out.loss.item())


def get_mismatch_token_indices(clean_text: str, typo_text: str, tokenizer) -> list:
    """
    Token-level difflib approach:
    1) tokenize both
    2) use difflib.SequenceMatcher to find mismatch ranges
    3) return the mismatch token indices in the 'typo' token sequence
    """
    clean_tokens = tokenizer.tokenize(clean_text)
    typo_tokens  = tokenizer.tokenize(typo_text)

    sm = difflib.SequenceMatcher(None, clean_tokens, typo_tokens)
    mismatch_indices = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            # tokens [j1:j2] in the typo sequence differ
            mismatch_indices.extend(range(j1, j2))
    return mismatch_indices


def main():
    logger.info("=== Starting token_level_hook_difflib.py ===")

    # 1) Load model & tokenizer
    logger.info(f"Loading model from {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")

    # 2) Load PC1
    if not os.path.exists(PC1_VECTORS_PATH):
        logger.error(f"{PC1_VECTORS_PATH} not found!")
        return
    data = torch.load(PC1_VECTORS_PATH, map_location="cpu")
    if PC1_LAYER_KEY not in data:
        logger.error(f"No key '{PC1_LAYER_KEY}' in {list(data.keys())}")
        return
    pc1_raw = data[PC1_LAYER_KEY]
    if isinstance(pc1_raw, np.ndarray):
        pc1_vector = torch.from_numpy(pc1_raw).float()
    else:
        pc1_vector = pc1_raw.float()
    pc1_vector = pc1_vector.to(DEVICE)
    pc1_norm_sq = torch.sum(pc1_vector * pc1_vector).item()
    logger.info(f"PC1 loaded for {PC1_LAYER_KEY}, shape={pc1_vector.shape}, norm={pc1_norm_sq**0.5:.4f}")

    # 3) Define hooking function that only removes dimension at mismatch indices
    def token_level_remove_fn(module, input_, output_):
        """
        We'll store mismatch indices in 'module._mismatch_indices'.
        Only subtract the dimension for those token positions.
        """
        pc1_ = pc1_vector.unsqueeze(0).unsqueeze(0)
        dot = (output_ * pc1_).sum(dim=-1, keepdim=True)
        proj = dot / pc1_norm_sq * pc1_

        bs, seq_len, hidden_dim = output_.shape
        token_mask = torch.zeros(seq_len, device=output_.device, dtype=output_.dtype)
        if hasattr(module, "_mismatch_indices") and module._mismatch_indices:
            for idx in module._mismatch_indices:
                if idx < seq_len:
                    token_mask[idx] = 1.0
        token_mask_3d = token_mask.unsqueeze(0).unsqueeze(-1)
        return output_ - proj * token_mask_3d

    layer_module = model.model.layers[LAYER_INDEX].mlp
    logger.info(f"Hook target: {layer_module}")

    def generate_text(prompt: str, mismatch_indices=None):
        """
        We store mismatch_indices in the module, register the forward hook, do generation.
        """
        # store mismatch indices
        layer_module._mismatch_indices = mismatch_indices or []
        hook = layer_module.register_forward_hook(token_level_remove_fn)

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

        hook.remove()
        layer_module._mismatch_indices = None
        return text

    # 4) read parallel CSV with columns [clean, typo]
    if not os.path.exists(PARALLEL_CSV):
        logger.error(f"{PARALLEL_CSV} not found!")
        return
    lines = []
    with open(PARALLEL_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                c = row[0].strip()
                t = row[1].strip()
                lines.append((c, t))

    if SUBSAMPLE_LINES is not None and len(lines) > SUBSAMPLE_LINES:
        lines = lines[:SUBSAMPLE_LINES]
    logger.info(f"Loaded {len(lines)} parallel (clean, typo) pairs from {PARALLEL_CSV}")

    out_header = [
        "clean_prompt", "typo_prompt",
        "text_normal", "ppl_normal",
        "text_tokenRemove", "ppl_tokenRemove"
    ]
    out_rows = [out_header]

    # 5) main loop
    logger.info("Starting token-level hooking per mismatch.")
    for i, (clean_text, typo_text) in enumerate(tqdm(lines, desc="TokenLevelDifflib")):
        if i % 50 == 0 and i > 0:
            logger.info(f"Processed {i} / {len(lines)} lines...")

        # find mismatch indices
        mismatch_indices = get_mismatch_token_indices_pair(clean_text, typo_text, tokenizer)
        # normal
        text_normal = None
        ppl_normal  = None
        enc = tokenizer.encode(typo_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
                                 temperature=TEMPERATURE, top_p=TOP_P)
        text_normal = tokenizer.decode(out[0], skip_special_tokens=True)
        ppl_normal  = compute_perplexity(model, tokenizer, text_normal)

        # token-level remove
        text_remove = generate_text(typo_text, mismatch_indices)
        ppl_remove  = compute_perplexity(model, tokenizer, text_remove)

        row = [clean_text, typo_text,
               text_normal, f"{ppl_normal:.3f}",
               text_remove, f"{ppl_remove:.3f}"]
        out_rows.append(row)

    logger.info(f"Done generating. Writing {len(out_rows)-1} lines to {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    logger.info("=== Completed token_level_hook_difflib.py ===")


def get_mismatch_token_indices_pair(clean_text, typo_text, tokenizer):
    """
    The actual difflib approach. 
    Returns the mismatch token indices in 'typo_text' token space.
    """
    clean_tokens = tokenizer.tokenize(clean_text)
    typo_tokens  = tokenizer.tokenize(typo_text)

    sm = difflib.SequenceMatcher(None, clean_tokens, typo_tokens)
    mismatch_indices = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            mismatch_indices.extend(range(j1, j2))
    return mismatch_indices

if __name__ == "__main__":
    main()
