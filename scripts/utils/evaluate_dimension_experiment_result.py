#!/usr/bin/env python3

"""
Reads outputs from the four specialized scripts:
  1) token_level_hook_difflib.py
  2) typo_correction_experiment.py
  3) next_token_distribution_difflib.py
  4) typo_classification_experiment.py

Each produces its own CSV with unique columns.

We define global paths/toggles for each file, parse them if they exist,
and compute basic stats (like average perplexities, classification accuracy, 
differences between normal vs. remove-dim, etc.).

Results are logged to "analyze_all_specialized_results.log".
You can adapt or expand the analysis as needed.
"""

import os
import csv
import math
import logging
from statistics import mean, median
from typing import List, Optional

# -----------------------
# GLOBAL CONFIG
# -----------------------

# File paths from each specialized script:
TOKEN_LEVEL_HOOK_FILE  = "analyses_results/token_level_hook_difflib_results.csv"
TYPO_CORRECTION_FILE   = "analyses_results/typo_correction_exp_results.csv"
NEXT_TOKEN_DIST_FILE   = "analyses_results/next_token_distribution_difflib.csv"
TYPO_CLASSIFY_FILE     = "analyses_results/typo_classification_results.csv"

# Toggles: set True/False to parse each file
ANALYZE_TOKEN_LEVEL  = True
ANALYZE_CORRECTION   = True
ANALYZE_NEXT_DIST    = True
ANALYZE_CLASSIFY     = True

LOG_FILENAME         = "logs/analyze_all_specialized_results.log"

# -----------------------
# LOGGING SETUP
# -----------------------
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_token_level_hook_file(csv_path: str):
    """
    Expects columns:
      clean_prompt, typo_prompt,
      text_normal, ppl_normal,
      text_tokenRemove, ppl_tokenRemove

    We'll parse the perplexities, compute average difference, etc.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"[Token-Level Hook] File not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            logger.warning("[Token-Level Hook] CSV is empty.")
            return

        # find columns
        try:
            idx_ppl_normal = header.index("ppl_normal")
            idx_ppl_remove = header.index("ppl_tokenRemove")
        except ValueError:
            logger.warning(f"[Token-Level Hook] Required columns not found in header: {header}")
            return

        normal_vals = []
        remove_vals = []
        num_rows = 0
        for row in reader:
            if len(row) <= max(idx_ppl_normal, idx_ppl_remove):
                continue
            try:
                n_val = float(row[idx_ppl_normal])
                r_val = float(row[idx_ppl_remove])
            except ValueError:
                continue
            normal_vals.append(n_val)
            remove_vals.append(r_val)
            num_rows += 1

        if num_rows == 0:
            logger.warning("[Token-Level Hook] No valid rows parsed.")
            return

        avg_norm = mean(normal_vals)
        med_norm = median(normal_vals)
        avg_rem  = mean(remove_vals)
        med_rem  = median(remove_vals)
        diffs = [rv - nv for (rv, nv) in zip(remove_vals, normal_vals)]
        avg_diff = mean(diffs)
        med_diff = median(diffs)

        logger.info("=== [Token-Level Hook] Stats ===")
        logger.info(f"  Found {num_rows} rows in {csv_path}")
        logger.info(f"  Normal perplexity:    mean={avg_norm:.3f}, median={med_norm:.3f}")
        logger.info(f"  TokenRemove perplexity: mean={avg_rem:.3f}, median={med_rem:.3f}")
        logger.info(f"  (remove - normal):    mean={avg_diff:+.3f}, median={med_diff:+.3f}")


def parse_typo_correction_file(csv_path: str):
    """
    Expects columns (by default):
      original_prompt,
      correct_prompt,  ppl_normal,
      correct_removeDim, ppl_removeDim
    optionally correct_addDim, ppl_addDim if add test was used
    """
    if not os.path.exists(csv_path):
        logger.warning(f"[Typo Correction] File not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            logger.warning("[Typo Correction] CSV is empty.")
            return

        # find columns
        try:
            idx_ppl_normal = header.index("ppl_normal")
            idx_ppl_remove = header.index("ppl_removeDim")
        except ValueError:
            logger.warning(f"[Typo Correction] Required columns not found in header: {header}")
            return

        # optional add
        idx_ppl_add = None
        if "ppl_addDim" in header:
            idx_ppl_add = header.index("ppl_addDim")

        normal_vals = []
        remove_vals = []
        add_vals    = []

        num_rows = 0
        for row in reader:
            if len(row) <= idx_ppl_remove:
                continue
            try:
                n_val = float(row[idx_ppl_normal])
                r_val = float(row[idx_ppl_remove])
            except ValueError:
                continue
            normal_vals.append(n_val)
            remove_vals.append(r_val)

            if idx_ppl_add is not None and len(row) > idx_ppl_add:
                try:
                    a_val = float(row[idx_ppl_add])
                    add_vals.append(a_val)
                except ValueError:
                    pass

            num_rows += 1

        if num_rows == 0:
            logger.warning("[Typo Correction] No valid rows parsed.")
            return

        avg_norm = mean(normal_vals)
        med_norm = median(normal_vals)
        avg_rem  = mean(remove_vals)
        med_rem  = median(remove_vals)
        diffs_remove = [rv - nv for (rv,nv) in zip(remove_vals, normal_vals)]
        avg_diff_rem = mean(diffs_remove)
        med_diff_rem = median(diffs_remove)

        logger.info("=== [Typo Correction] Stats ===")
        logger.info(f"  Found {num_rows} rows in {csv_path}")
        logger.info(f"  Normal perplexity:   mean={avg_norm:.3f}, median={med_norm:.3f}")
        logger.info(f"  Remove perplexity:   mean={avg_rem:.3f}, median={med_rem:.3f}")
        logger.info(f"  remove - normal:     mean={avg_diff_rem:+.3f}, median={med_diff_rem:+.3f}")

        if add_vals:
            avg_add = mean(add_vals)
            med_add = median(add_vals)
            diffs_add = [a - n for (a,n) in zip(add_vals, normal_vals)]
            avg_diff_add = mean(diffs_add)
            med_diff_add = median(diffs_add)
            logger.info("  [Add-Dim Present]")
            logger.info(f"   add perplexity:    mean={avg_add:.3f}, median={med_add:.3f}")
            logger.info(f"   add - normal:      mean={avg_diff_add:+.3f}, median={med_diff_add:+.3f}")


def parse_next_token_distribution_file(csv_path: str):
    """
    Expects columns:
     clean_prompt, typo_prompt, slip_index,
     top_normal_tokens, top_normal_probs,
     top_remove_tokens, top_remove_probs

    We'll parse each row, but there's no single numeric perplexity to average.
    We can do a 'ranking difference' or just count how often top-1 token changes.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"[Next-Token Dist] File not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            logger.warning("[Next-Token Dist] CSV is empty.")
            return

        # find columns
        try:
            idx_norm_toks = header.index("top_normal_tokens")
            idx_norm_probs= header.index("top_normal_probs")
            idx_rem_toks  = header.index("top_remove_tokens")
            idx_rem_probs = header.index("top_remove_probs")
        except ValueError:
            logger.warning("[Next-Token Dist] Required columns not found. Check header.")
            return

        num_rows = 0
        changed_top1_count = 0
        changed_any_count  = 0

        for row in reader:
            if len(row) <= idx_rem_probs:
                continue

            normal_tokens = row[idx_norm_toks].split()
            remove_tokens = row[idx_rem_toks].split()
            if not normal_tokens or not remove_tokens:
                continue
            num_rows += 1

            # check if top1 differs
            if normal_tokens[0] != remove_tokens[0]:
                changed_top1_count += 1

            # check if there's any difference in top-K sets
            norm_set = set(normal_tokens)
            remv_set = set(remove_tokens)
            if norm_set != remv_set:
                changed_any_count += 1

        if num_rows == 0:
            logger.warning("[Next-Token Dist] No valid rows parsed.")
            return

        frac_changed_top1 = changed_top1_count / num_rows
        frac_changed_any  = changed_any_count  / num_rows

        logger.info("=== [Next-Token Distribution] Stats ===")
        logger.info(f"  Found {num_rows} rows in {csv_path}")
        logger.info(f"  fraction with changed top1 token: {frac_changed_top1:.2%}")
        logger.info(f"  fraction with changed top-K set:  {frac_changed_any:.2%}")


def parse_typo_classification_file(csv_path: str):
    """
    Expects columns:
      original_prompt, class_normal, ppl_normal, class_removeDim, ppl_removeDim
    optional class_addDim, ppl_addDim if add test used.

    We'll measure how often normal says "YES"/"NO", remove says "YES"/"NO".
    Possibly compute 'accuracy' if we have a label. 
    For now, let's see how often classification changes from normal to remove. 
    """
    if not os.path.exists(csv_path):
        logger.warning(f"[Typo Classification] File not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            logger.warning("[Typo Classification] CSV empty.")
            return

        # columns we want
        try:
            idx_class_normal = header.index("class_normal")
            idx_class_remove = header.index("class_removeDim")
        except ValueError:
            logger.warning(f"[Typo Classification] Missing normal/remove classification columns in {header}")
            return

        # optional add
        idx_class_add = None
        if "class_addDim" in header:
            idx_class_add = header.index("class_addDim")

        # We'll count how often normal vs. remove differ
        normal_yes, normal_no, normal_qq = 0,0,0
        remove_yes, remove_no, remove_qq = 0,0,0
        changed_count = 0
        num_rows = 0

        add_yes, add_no, add_qq = 0,0,0
        changed_count_add = 0

        for row in reader:
            if len(row) <= idx_class_remove:
                continue
            c_norm = row[idx_class_normal].strip().upper()
            c_rem  = row[idx_class_remove].strip().upper()
            num_rows += 1

            # normal
            if c_norm == "YES":
                normal_yes += 1
            elif c_norm == "NO":
                normal_no += 1
            else:
                normal_qq += 1

            # remove
            if c_rem == "YES":
                remove_yes += 1
            elif c_rem == "NO":
                remove_no += 1
            else:
                remove_qq += 1

            if c_norm != c_rem:
                changed_count += 1

            # optional add
            if idx_class_add is not None and len(row) > idx_class_add:
                c_add = row[idx_class_add].strip().upper()
                if c_add == "YES":
                    add_yes += 1
                elif c_add == "NO":
                    add_no += 1
                else:
                    add_qq += 1

                if c_norm != c_add:
                    changed_count_add += 1

        if num_rows == 0:
            logger.warning("[Typo Classification] No valid rows.")
            return

        logger.info("=== [Typo Classification] Stats ===")
        logger.info(f"  Found {num_rows} rows in {csv_path}")
        logger.info(f"  normal:  YES={normal_yes}, NO={normal_no}, ??={normal_qq}")
        logger.info(f"  remove:  YES={remove_yes}, NO={remove_no}, ??={remove_qq}")
        logger.info(f"  changed normal->remove: {changed_count} / {num_rows}  ({changed_count/num_rows:.2%})")

        if idx_class_add is not None:
            logger.info(f"  add: YES={add_yes}, NO={add_no}, ??={add_qq}")
            logger.info(f"  changed normal->add: {changed_count_add} / {num_rows}  ({changed_count_add/num_rows:.2%})")


def main():
    logger.info("=== Starting analyze_all_specialized_results.py ===")

    if ANALYZE_TOKEN_LEVEL:
        logger.info(f"[token_level_hook] Attempting parse: {TOKEN_LEVEL_HOOK_FILE}")
        parse_token_level_hook_file(TOKEN_LEVEL_HOOK_FILE)

    if ANALYZE_CORRECTION:
        logger.info(f"[typo_correction] Attempting parse: {TYPO_CORRECTION_FILE}")
        parse_typo_correction_file(TYPO_CORRECTION_FILE)

    if ANALYZE_NEXT_DIST:
        logger.info(f"[next_token_distribution] Attempting parse: {NEXT_TOKEN_DIST_FILE}")
        parse_next_token_distribution_file(NEXT_TOKEN_DIST_FILE)

    if ANALYZE_CLASSIFY:
        logger.info(f"[typo_classification] Attempting parse: {TYPO_CLASSIFY_FILE}")
        parse_typo_classification_file(TYPO_CLASSIFY_FILE)

    logger.info("=== Completed analyze_all_specialized_results.py ===")


if __name__ == "__main__":
    main()
