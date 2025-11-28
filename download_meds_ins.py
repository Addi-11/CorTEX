"""
Download Henrychur/MedS-Ins dataset from HuggingFace
Handles JSON parsing issues by downloading raw files directly
"""

import os
import json
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

DATASET_REPO = "Henrychur/MedS-Ins"
OUTPUT_DIR = "datasets/MedS-Ins"

def download_dataset():
    """Download all files from the dataset repository."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Listing files in {DATASET_REPO}...")
    
    files = list_repo_files(DATASET_REPO, repo_type="dataset")
    json_files = [f for f in files if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files")
    
    downloaded = []
    failed = []
    
    for filename in tqdm(json_files, desc="Downloading files"):
        try:
            local_path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=OUTPUT_DIR,
                local_dir_use_symlinks=False
            )
            downloaded.append(filename)
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")
            failed.append(filename)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(downloaded)} files")
    print(f"Failed: {len(failed)} files")
    print(f"Files saved to: {OUTPUT_DIR}")
    
    print(f"\nDownloaded files:")
    for f in sorted(downloaded)[:20]:
        print(f"  - {f}")
    if len(downloaded) > 20:
        print(f"  ... and {len(downloaded) - 20} more")
    
    return downloaded, failed


def load_json_file_safe(filepath):
    """
    Safely load a JSON file, handling both JSON arrays and JSONL formats.
    """
    data = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
                return data
            elif content.startswith('{'):
                pass
    except json.JSONDecodeError:
        pass
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except json.JSONDecodeError:
                        continue
        if data:
            return data
    except Exception:
        pass
    
    return None


if __name__ == "__main__":
    downloaded, failed = download_dataset()