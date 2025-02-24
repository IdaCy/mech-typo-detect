import re

# Define paths
raw_clean_file = "prompts/raw/cleanQs.csv"
raw_typo_file = "prompts/raw/typoQs.csv"
output_clean_issues_file = "tests/cleanQs_issues.csv"
output_typo_issues_file = "tests/typoQs_issues.csv"

# Read the file line by line to ensure correct sentence parsing
with open(raw_clean_file, "r", encoding="utf-8") as f:
    clean_sentences = [line.strip() for line in f.readlines() if line.strip()]

with open(raw_typo_file, "r", encoding="utf-8") as f:
    typo_sentences = [line.strip() for line in f.readlines() if line.strip()]

# Define issue detection function
def detect_issues(sentence):
    issues = []

    if not isinstance(sentence, str) or len(sentence.strip()) == 0:  # Empty or non-string
        issues.append("Empty sentence")

    if len(sentence.split()) < 3:  # Sentence too short
        issues.append("Sentence too short")

    if len(sentence.split()) > 400:  # Sentence unusually long
        issues.append("Sentence unusually long")

    if re.match(r'^[a-z]', sentence):  # Starts with lowercase letter
        issues.append("Sentence starts with lowercase")

    if sentence.isupper():  # Entirely uppercase
        issues.append("Sentence in all caps")

    if sentence.endswith((" ", "\t")) or sentence.startswith((" ", "\t")):  # Leading/trailing spaces
        issues.append("Leading/trailing spaces")

    if re.search(r'\s{2,}', sentence):  # Extra spaces in the middle
        issues.append("Extra spaces")

    #if re.search(r'[^\w\s.,?!\'"-]', sentence):  # Strange characters (only allow standard text)
    #    issues.append("Unusual characters")

    if re.search(r'(?!\.\.\.)[?!.,]{3,}', sentence):  # Too many punctuation marks in a row
        issues.append("Excessive punctuation")

    if not re.search(r'[.?!]$', sentence):  # Missing proper ending punctuation
        issues.append("No punctuation at end")

#    regex_pattern = r'[^\w\s.,?!\'"()\-\—:;]'
#    regex_pattern = r'[^\w\s.,?!\'"()\[\]{}<>:;\-\—/_%&*=+@#$]'
    regex_pattern = r'[^\w\s.,?!\'"()\[\]{}<>:;\-\—/_%&*=+@#$πµΩ]'
    if re.search(regex_pattern, sentence):
        issues.append("Misplaced commas")

    if re.search(r'""', sentence):  # Double double quotes ("")
        issues.append("Double double quotes")

    return issues

# Apply issue detection
clean_issues = [(sentence, detect_issues(sentence)) for sentence in clean_sentences if detect_issues(sentence)]
typo_issues = [(sentence, detect_issues(sentence)) for sentence in typo_sentences if detect_issues(sentence)]

# Save issue reports
with open(output_clean_issues_file, "w", encoding="utf-8") as f:
    for sentence, issues in clean_issues:
        f.write(f"{sentence}\t[{', '.join(issues)}]\n")

with open(output_typo_issues_file, "w", encoding="utf-8") as f:
    for sentence, issues in typo_issues:
        f.write(f"{sentence}\t[{', '.join(issues)}]\n")

# Print summary
print("\nProblematic Sentences in Clean Dataset:")
for sentence, issues in clean_issues[:10]:
    print(f"- {sentence}  [Issue: {', '.join(issues)}]")

print("\nProblematic Sentences in Typo Dataset:")
for sentence, issues in typo_issues[:10]:
    print(f"- {sentence}  [Issue: {', '.join(issues)}]")

print("\nProblematic sentences saved to:")
print(f"{output_clean_issues_file}")
print(f"{output_typo_issues_file}")
