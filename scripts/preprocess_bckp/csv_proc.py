import os
import pandas as pd
import re

# Define paths for raw input files
raw_clean_file = "data/raw/cleanQs.csv"
raw_typo_file  = "data/raw/typoQs.csv"

# Define paths for preprocessed output files
preprocessed_clean_file = "data/preprocessed/cleanQs9.csv"
preprocessed_typo_file  = "data/preprocessed/typoQs9.csv"

# Ensure the output directory exists
os.makedirs("data/preprocessed", exist_ok=True)

# Read CSV as a single column without breaking on commas
df_clean = pd.read_csv(raw_clean_file, header=0, dtype=str, usecols=[0], encoding="utf-8", keep_default_na=False)
df_typo = pd.read_csv(raw_typo_file, header=0, dtype=str, usecols=[0], encoding="utf-8", keep_default_na=False)

# Debugging: Print file line counts before/after reading
print(f"Original clean file: {len(pd.read_csv(raw_clean_file, header=0, dtype=str))} lines")
print(f"After reading: {len(df_clean)} lines")

print(f"Original typo file: {len(pd.read_csv(raw_typo_file, header=0, dtype=str))} lines")
print(f"After reading: {len(df_typo)} lines")

# Function to properly remove all leading/trailing quotes without breaking text
def clean_text(text):
    if isinstance(text, str):
        text = text.strip()  # Remove extra spaces
        text = re.sub(r'^[\'"]+|[\'"]+$', '', text)  # Remove leading/trailing quotes
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces (avoid issues from stripping)
    return text

# Apply text cleaning before saving
df_clean.iloc[:, 0] = df_clean.iloc[:, 0].apply(clean_text)
df_typo.iloc[:, 0] = df_typo.iloc[:, 0].apply(clean_text)

# Save the cleaned data (NO escape characters, NO extra quotes)
df_clean.to_csv(preprocessed_clean_file, index=False, header=False, encoding="utf-8", quoting=0, escapechar='\\')
df_typo.to_csv(preprocessed_typo_file, index=False, header=False, encoding="utf-8", quoting=0, escapechar='\\')

print("\nPreprocessing complete! Sentences fully preserved & quotes removed.")
print(f"Clean data saved to: {preprocessed_clean_file}")
print(f"Typo data saved to: {preprocessed_typo_file}")
