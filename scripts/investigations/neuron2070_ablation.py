#!/usr/bin/env python3

"""
neuron2070_ablation.py

Demonstration script that specifically zeros out neuron #2070's activation
at a chosen layer (layer_2 by default). Then we compare normal generation
vs. "only that neuron suppressed" generation, measuring perplexities
and final text.

Global config at top. 
"""

import os
import csv
import math
import torch
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------
# GLOBAL CONFIG
# -----------------------
DATASET_CSV       = "prompts/preprocessed/typoQs.csv"
SUBSAMPLE_LINES   = 200
MODEL_NAME        = "../typo-correct-subspaces/models/mistral-7b"
LAYER_INDEX       = 2
NEURON_INDEX      = 2070
DEVICE            = "cuda"

OUTPUT_CSV        = "analyses_results/neuron2070_ablation.csv"
LOG_FILE          = "neuron2070_ablation.log"

MAX_NEW_TOKENS    = 40
DO_SAMPLE         = True
TEMPERATURE       = 0.7
TOP_P             = 0.9
# -----------------------

logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_perplexity(model, tokenizer, text: str):
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())

def main():
    logger.info("=== Starting neuron2070_ablation.py ===")
    logger.info(f"LAYER_INDEX={LAYER_INDEX}, NEURON_INDEX={NEURON_INDEX}")

    # 1) load model & tokenizer
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    # We'll define a hook that sets output_[:, :, NEURON_INDEX] = 0
    def neuron_suppression_hook_fn(module, input_, output_):
        # shape: (batch, seq_len, hidden_dim)
        # We forcibly zero out the NEURON_INDEX dimension - in-place assignment to output_:
        output_[:,:,NEURON_INDEX] = 0
        return output_

    # get reference to the layer
    layer_module = model.model.layers[LAYER_INDEX].mlp
    logger.info(f"Hook target = model.model.layers[{LAYER_INDEX}].mlp => {layer_module}")

    def generate_text(prompt, hook=False):
        handle = None
        if hook:
            handle = layer_module.register_forward_hook(neuron_suppression_hook_fn)
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
        if handle:
            handle.remove()
        return text

    # 2) read CSV lines
    if not os.path.exists(DATASET_CSV):
        logger.error(f"{DATASET_CSV} not found!")
        return
    lines = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                lines.append(row[0].strip())

    if SUBSAMPLE_LINES is not None and len(lines) > SUBSAMPLE_LINES:
        lines = lines[:SUBSAMPLE_LINES]
    logger.info(f"Loaded {len(lines)} lines from {DATASET_CSV}")

    out_header = ["prompt", "text_normal", "ppl_normal", "text_neuronAblated", "ppl_neuronAblated"]
    out_rows = [out_header]

    logger.info("Starting ablation generation loop...")

    for i, line in enumerate(tqdm(lines, desc="Neuron2070Ablation")):
        if i % 50 == 0 and i>0:
            logger.info(f"Processed {i} lines...")

        text_normal = generate_text(line, hook=False)
        ppl_normal  = compute_perplexity(model, tokenizer, text_normal)

        text_supp   = generate_text(line, hook=True)
        ppl_supp    = compute_perplexity(model, tokenizer, text_supp)

        row = [line, text_normal, f"{ppl_normal:.3f}", text_supp, f"{ppl_supp:.3f}"]
        out_rows.append(row)

    logger.info(f"Done. Writing {len(out_rows)-1} lines to {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    logger.info("=== Completed neuron2070_ablation.py ===")

if __name__ == "__main__":
    main()
