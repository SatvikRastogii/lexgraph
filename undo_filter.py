import os
import json
import shutil
from collections import defaultdict

def undo_and_extract_40_cases():
    corpus_dir = r"c:\Users\Satvik Rastogi\Downloads\graphrag-project\legal_corpus"
    input_dir = r"c:\Users\Satvik Rastogi\Downloads\graphrag-project\input"
    metadata_path = r"c:\Users\Satvik Rastogi\Downloads\graphrag-project\corpus_metadata.json"
    
    # Empty input directory completely
    for f in os.listdir(input_dir):
        file_path = os.path.join(input_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    print("Wiped input/ directory to start fresh.")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Group files by their article focus
    categories = defaultdict(list)
    for doc in metadata.get('documents', []):
        cat = doc.get('article_focus', 'Unknown')
        filename = doc.get('filename')
        categories[cat].append(filename)

    print(f"Found categories: {list(categories.keys())}")

    # Select exactly 40 cases (we will take a balanced mix from each category if possible)
    # We want 40 total. Let's say 8 from each of 5 categories
    target_count = 40
    selected_files = []
    
    # Try to grab an equal chunk from each category
    per_cat = target_count // len(categories) if categories else target_count
    for cat, files in categories.items():
        selected_files.extend(files[:per_cat])
    
    # Fill remaining required slots if any category didn't have enough
    remaining_needed = target_count - len(selected_files)
    if remaining_needed > 0:
        all_remaining_files = []
        for cat, files in categories.items():
            all_remaining_files.extend(files[per_cat:])
        
        selected_files.extend(all_remaining_files[:remaining_needed])
        
    copied = 0
    for filename in selected_files:
        source = os.path.join(corpus_dir, filename)
        dest = os.path.join(input_dir, filename)
        if os.path.exists(source):
            shutil.copy2(source, dest)
            copied += 1

    print(f"\nRestored and extracted {copied} cases to input/!")

if __name__ == "__main__":
    undo_and_extract_40_cases()
