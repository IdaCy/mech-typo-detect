#!/usr/bin/env python3

"""
A script to run on either a "typo" dataset or a "clean" dataset, set by DATASET_CHOICE.
We do:
1) Normal generation (no hook).
2) Dimension-removed generation (subtracting projection onto PC1).
3) (Optionally) dimension-added generation if PERFORM_ADD_TEST=True.

All logs are written to a separate file: 'my_causal_experiment.log'
so you can track progress in detail.
"""

import os
import csv
import math
import torch
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# SCOPE/SPEED CONFIG
DATASET_CHOICE   = "typo"   # or "typo"
SUBSAMPLE_LINES  = 2000      # Number of lines to process from CSV (None = use all)
MAX_NEW_TOKENS   = 50        # Fewer tokens => faster generation
PERFORM_ADD_TEST = False     # If True, do dimension-add hooking (3rd run) => slower

# MODEL & FILE CONFIG
MODEL_NAME       = "../typo-correct-subspaces/models/mistral-7b"
PC1_VECTORS_PATH = "analyses_results/PCA_PC1/layer_pc1_vectors.pt"
PC1_LAYER_KEY    = "layer_2"
LAYER_INDEX      = 2
DEVICE           = "cuda"

# For "typo" or "clean"
INPUT_FILE_TYPO  = "prompts/preprocessed/typoQs.csv"
INPUT_FILE_CLEAN = "prompts/preprocessed/cleanQs.csv"
OUTPUT_CSV_FILE  = "analyses_results/dimension_causal_S_" + DATASET_CHOICE + ".csv"

# Generation sampling config
DO_SAMPLE   = True
TEMPERATURE = 0.7
TOP_P       = 0.9

# -----------------------
# LOGGING SETUP
# -----------------------
LOG_FILENAME = "logs/S_" + DATASET_CHOICE + "_causal_experiment.log"
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='w',  # overwrite each run, or 'a' to append
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of the final text under the model's distribution."""
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())

def main():
    logger.info("===== STARTING run_dimension_causal_experiment_single.py =====")

    # 1) Decide which file to read
    if DATASET_CHOICE == "typo":
        INPUT_FILE = INPUT_FILE_TYPO
    elif DATASET_CHOICE == "clean":
        INPUT_FILE = INPUT_FILE_CLEAN
    else:
        raise ValueError(f"Unknown DATASET_CHOICE: {DATASET_CHOICE}")

    logger.info(f"DATASET_CHOICE={DATASET_CHOICE}, using INPUT_FILE={INPUT_FILE}")
    logger.info(f"SUBSAMPLE_LINES={SUBSAMPLE_LINES}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, PERFORM_ADD_TEST={PERFORM_ADD_TEST}")

    # 2) Load model & tokenizer
    logger.info(f"Loading model & tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    logger.info("Model & tokenizer loaded successfully.")

    # 3) Load PC1 vector
    if not os.path.exists(PC1_VECTORS_PATH):
        raise FileNotFoundError(f"{PC1_VECTORS_PATH} not found!")
    pca_data = torch.load(PC1_VECTORS_PATH, map_location="cpu")
    logger.info(f"Loaded PCA dictionary from {PC1_VECTORS_PATH}. Keys: {list(pca_data.keys())}")

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
    logger.info(f"PC1 vector for {PC1_LAYER_KEY} loaded. shape={pc1_vector.shape}, layer={LAYER_INDEX}")

    pc1_norm_sq = torch.sum(pc1_vector * pc1_vector).item()
    logger.info(f"PC1 norm squared = {pc1_norm_sq:.4f}")

    # 4) Define hooking
    def remove_dim_hook_fn(module, input_, output_):
        # dimension removal => subtract projection on pc1_vector
        pc1_ = pc1_vector.unsqueeze(0).unsqueeze(0)
        dot = (output_ * pc1_).sum(dim=-1, keepdim=True)
        proj = dot / pc1_norm_sq * pc1_
        return output_ - proj

    def add_dim_hook_fn(module, input_, output_):
        # forcibly add pc1
        pc1_ = pc1_vector.unsqueeze(0).unsqueeze(0)
        return output_ + pc1_

    layer_module = model.model.layers[LAYER_INDEX].mlp
    logger.info(f"Hook target = model.model.layers[{LAYER_INDEX}].mlp => {layer_module}")

    def generate_text(prompt: str, hook=None):
        if hook is not None:
            handle = layer_module.register_forward_hook(hook)
            logger.debug(f"Registered forward hook: {hook.__name__} for prompt: {prompt[:60]}...")
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
            logger.debug("Removed forward hook.")
        return text

    # 5) Read prompts
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found!")
    lines = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                lines.append(row[0].strip())

    logger.info(f"Loaded {len(lines)} lines from {INPUT_FILE}.")
    if SUBSAMPLE_LINES is not None and len(lines) > SUBSAMPLE_LINES:
        lines = lines[:SUBSAMPLE_LINES]
        logger.info(f"Subsampled to {len(lines)} lines.")

    # 6) Prepare CSV output
    out_header = ["prompt", "text_normal", "ppl_normal", "text_removeDim", "ppl_removeDim"]
    if PERFORM_ADD_TEST:
        out_header += ["text_addDim", "ppl_addDim"]

    out_rows = [out_header]
    logger.info("Starting main loop over prompts...")

    # 7) For each prompt, run generation
    for i, prompt in enumerate(tqdm(lines, desc=f"{DATASET_CHOICE.capitalize()}Exp")):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processing line {i} / {len(lines)}...")

        # normal
        text_normal = generate_text(prompt, None)
        ppl_normal  = compute_perplexity(model, tokenizer, text_normal)
        logger.debug(f"Prompt idx={i}, normal PPL={ppl_normal:.3f}")

        # remove
        text_remove = generate_text(prompt, remove_dim_hook_fn)
        ppl_remove  = compute_perplexity(model, tokenizer, text_remove)
        logger.debug(f"Prompt idx={i}, removeDim PPL={ppl_remove:.3f}")

        row = [prompt, text_normal, f"{ppl_normal:.3f}", text_remove, f"{ppl_remove:.3f}"]

        # (optional) add
        if PERFORM_ADD_TEST:
            text_add = generate_text(prompt, add_dim_hook_fn)
            ppl_add  = compute_perplexity(model, tokenizer, text_add)
            logger.debug(f"Prompt idx={i}, addDim PPL={ppl_add:.3f}")
            row += [text_add, f"{ppl_add:.3f}"]

        out_rows.append(row)

    logger.info("Finished generating all prompts.")
    logger.info(f"Writing {len(out_rows)-1} results to {OUTPUT_CSV_FILE}")

    # 8) Write output
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    with open(OUTPUT_CSV_FILE, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    logger.info("Done writing CSV.")
    logger.info(f"Last row: {out_rows[-1]}")
    logger.info("===== COMPLETED run_dimension_causal_experiment_single.py =====")

if __name__ == "__main__":
    main()