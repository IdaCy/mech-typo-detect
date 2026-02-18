import os
import torch
import pandas as pd
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

CLEAN_FILE = "/workspace/prompts/preprocessed/cleanQs.csv"
TYPO_FILE = "/workspace/prompts/preprocessed/typo_repeated.csv"
OUTPUT_DIR = "extractions/clean_repeated/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "../typo-correct-subspaces/models/mistral-7b"
BATCH_SIZE = 4
USE_BFLOAT16 = True  # Use bfloat16 for storage efficiency
MAX_SEQ_LENGTH = 512
TOP_K_LOGITS = 10  # Store only top-10 logits per token

# Layers to extract
EXTRACT_HIDDEN_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 31]
EXTRACT_ATTENTION_LAYERS = [10, 15, 20, 25, 30, 31]  # Attention stored only in higher layers
FINAL_LAYER = 31  # Logits + probabilities from final layer

# ------------------------------------------------------------------------
# Load Model and Tokenizer
# ------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="eager"
)
model.resize_token_embeddings(len(tokenizer))
model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------------------
# Load Sentences
# ------------------------------------------------------------------------
def load_sentences(file_path):
    """Reads a file line-by-line and returns a list of sentences."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

clean_texts = load_sentences(CLEAN_FILE)
typo_texts = load_sentences(TYPO_FILE)

if len(clean_texts) != len(typo_texts):
    raise ValueError("Mismatch between number of clean and typo prompts!")

print(f"Loaded {len(clean_texts)} samples for inference.")

# ------------------------------------------------------------------------
# Identify Relevant Token Indices
# ------------------------------------------------------------------------
def get_relevant_token_indices_pair(clean_text, typo_text, tokenizer, window=3):
    """Finds token positions that differ between clean and typo versions."""
    tokens_clean = tokenizer.tokenize(clean_text)
    tokens_typo = tokenizer.tokenize(typo_text)
    sm = difflib.SequenceMatcher(None, tokens_clean, tokens_typo)

    diff_indices_clean, diff_indices_typo = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            diff_indices_clean.extend(range(i1, i2))
            diff_indices_typo.extend(range(j1, j2))

    def expand_indices(indices, max_len):
        expanded = set()
        for idx in indices:
            start = max(0, idx - window)
            end = min(max_len, idx + window + 1)
            expanded.update(range(start, end))
        return sorted(expanded)

    return (
        expand_indices(diff_indices_clean, len(tokens_clean)),
        tokens_clean,
        expand_indices(diff_indices_typo, len(tokens_typo)),
        tokens_typo,
    )

# ------------------------------------------------------------------------
# Run Inference and Extract Features
# ------------------------------------------------------------------------
def capture_activations(text_batch, indices_batch):
    try:
        encodings = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to("cuda")
        attention_mask = encodings["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        batch_results = {}

        for i in range(len(text_batch)):
            relevant_indices = indices_batch[i] if indices_batch else list(range(len(input_ids[i])))

            sample_result = {}

            # Store hidden states (as bfloat16)
            sample_result["hidden_states"] = {
                f"layer_{l}": hidden_states[l][i][relevant_indices].to(torch.bfloat16).cpu()
                for l in EXTRACT_HIDDEN_LAYERS
            }

            # Store attention scores (as bfloat16)
            sample_result["attention_scores"] = {
                f"layer_{l}": attentions[l][i].mean(dim=0)[relevant_indices].to(torch.bfloat16).cpu()
                for l in EXTRACT_ATTENTION_LAYERS
            }

            # Store only final-layer top-k logits & probabilities (as bfloat16)
            top_logits, top_indices = torch.topk(logits[i][relevant_indices], k=TOP_K_LOGITS, dim=-1)
            top_probs = probabilities[i][relevant_indices].gather(-1, top_indices)

            sample_result["top_k_logits"] = {f"token_{t}": top_logits[j].to(torch.bfloat16).cpu() for j, t in enumerate(relevant_indices)}
            sample_result["top_k_probs"] = {f"token_{t}": top_probs[j].to(torch.bfloat16).cpu() for j, t in enumerate(relevant_indices)}
            sample_result["top_k_indices"] = {f"token_{t}": top_indices[j].to(torch.int16).cpu() for j, t in enumerate(relevant_indices)}

            # Store the actual predictions
            # Generate predictions autoregressively
            generated_ids = model.generate(
                input_ids=input_ids[i].unsqueeze(0),
                attention_mask=attention_mask[i].unsqueeze(0),
                max_new_tokens=20,
                do_sample=False  # Deterministic output
            )
            sample_result["predicted_text"] = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            sample_result["tokens"] = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().tolist())
            sample_result["original_text"] = text_batch[i]

            batch_results[i] = sample_result

        return batch_results

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

# ------------------------------------------------------------------------
# Run Inference on Clean & Typo Inputs + Save Results
# ------------------------------------------------------------------------
print("Starting inference and extraction of relevant activations for clean and typo prompts...")

for start_idx in range(0, len(clean_texts), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    batch_clean, batch_typo = clean_texts[start_idx:end_idx], typo_texts[start_idx:end_idx]

    indices_clean_batch, indices_typo_batch = [], []
    tokens_clean_batch, tokens_typo_batch = [], []
    
    for clean_txt, typo_txt in zip(batch_clean, batch_typo):
        rel_clean, tokens_clean, rel_typo, tokens_typo = get_relevant_token_indices_pair(clean_txt, typo_txt, tokenizer)
        indices_clean_batch.append(rel_clean)
        indices_typo_batch.append(rel_typo)
        tokens_clean_batch.append(tokens_clean)
        tokens_typo_batch.append(tokens_typo)

    activations_clean = capture_activations(batch_clean, indices_clean_batch)
    activations_typo = capture_activations(batch_typo, indices_typo_batch)

    # Saving + Info Prints
    if activations_clean and activations_typo:
        for i in range(len(batch_clean)):
            sample_idx = start_idx + i
            filename = os.path.join(OUTPUT_DIR, f"activations_{sample_idx:05d}.pt")
            torch.save({"clean": activations_clean[i], "typo": activations_typo[i]}, filename)
        print(f"Saved activations for samples {start_idx} to {end_idx}")

print(f"Inference complete. Results saved in '{OUTPUT_DIR}'.")
