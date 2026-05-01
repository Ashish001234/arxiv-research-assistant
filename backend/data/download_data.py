"""
Download the arXiv dataset from Kaggle using kagglehub.
Run this once to get the raw JSON file (~4GB).
Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json
"""
import os
import shutil
import kagglehub
from backend.config import DATA_DIR, ARXIV_RAW_PATH


def download_arxiv_dataset():
    print("Downloading arXiv dataset from Kaggle...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print(f"Downloaded to: {path}")

    # kagglehub downloads to a cache dir — find and move the snapshot file
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.endswith(".json"):
                src = os.path.join(root, fname)
                os.makedirs(DATA_DIR, exist_ok=True)
                dst = ARXIV_RAW_PATH
                if not os.path.exists(dst):
                    print(f"Copying {fname} → {dst}")
                    shutil.copy2(src, dst)
                else:
                    print(f"Already exists: {dst}")
                return dst

    raise FileNotFoundError("Could not find the arXiv JSON file in the downloaded dataset.")


if __name__ == "__main__":
    dst = download_arxiv_dataset()
    size_gb = os.path.getsize(dst) / 1e9
    print(f"Done. File size: {size_gb:.2f} GB")
