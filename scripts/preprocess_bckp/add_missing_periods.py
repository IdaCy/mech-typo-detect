import re

# Define paths
raw_clean_file = "data/raw/cleanQs.csv"
raw_typo_file = "data/raw/typoQs.csv"
output_clean_file = "data/preprocessed/cleanQs_fixed.csv"
output_typo_file = "data/preprocessed/typoQs_fixed.csv"

# Check + fix missing punctuation
def fix_missing_period(sentence):
    if isinstance(sentence, str):
        sentence = sentence.strip()  # Remove extra spaces
        if not re.search(r'[.?!]$', sentence):  # Check if it ends with . ? !
            return sentence + '.'  # Add missing period
    return sentence

# Read raw files as text and process line-by-line
with open(raw_clean_file, "r", encoding="utf-8") as f:
    clean_sentences = [fix_missing_period(line.strip()) for line in f.readlines()]

with open(raw_typo_file, "r", encoding="utf-8") as f:
    typo_sentences = [fix_missing_period(line.strip()) for line in f.readlines()]

# Save manually to avoid pandas adding quotes
with open(output_clean_file, "w", encoding="utf-8") as f:
    f.write("\n".join(clean_sentences) + "\n")

with open(output_typo_file, "w", encoding="utf-8") as f:
    f.write("\n".join(typo_sentences) + "\n")

print("\nSentences with missing punctuation have been fixed!")
print(f"Fixed Clean data saved to: {output_clean_file}")
print(f"Fixed Typo data saved to: {output_typo_file}")
