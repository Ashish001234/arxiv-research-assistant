"""
Filter arXiv dataset to CS papers and save as JSONL.
Run after download_data.py.
"""
import json
import os
from tqdm import tqdm
from backend.config import ARXIV_RAW_PATH, PAPERS_JSONL_PATH


def is_cs_paper(categories: str) -> bool:
    cats = categories.strip()
    return cats.startswith("cs.") or " cs." in cats


def load_and_filter_cs_papers(json_path: str, output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    skipped = 0

    with open(json_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Filtering CS papers"):
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            cats = paper.get("categories", "")
            if not is_cs_paper(cats):
                continue

            title = paper.get("title", "").strip().replace("\n", " ")
            abstract = paper.get("abstract", "").strip().replace("\n", " ")

            # skip papers with missing title or abstract
            if not title or not abstract:
                skipped += 1
                continue

            record = {
                "id": paper["id"],
                "title": title,
                "abstract": abstract,
                "authors": paper.get("authors", ""),
                "categories": cats,
                "update_date": paper.get("update_date", ""),
            }
            fout.write(json.dumps(record) + "\n")
            count += 1

    print(f"Kept {count:,} CS papers. Skipped {skipped:,} records.")
    return count


def load_papers_from_jsonl(jsonl_path: str) -> list[dict]:
    """Load the filtered JSONL into a list of paper dicts. Used at index time."""
    papers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading papers"):
            papers.append(json.loads(line))
    return papers


if __name__ == "__main__":
    if not os.path.exists(ARXIV_RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at {ARXIV_RAW_PATH}. "
            "Run backend/data/download_data.py first."
        )

    count = load_and_filter_cs_papers(ARXIV_RAW_PATH, PAPERS_JSONL_PATH)
    print(f"Saved {count:,} papers to {PAPERS_JSONL_PATH}")

    # Quick sanity check
    with open(PAPERS_JSONL_PATH, "r") as f:
        sample = json.loads(f.readline())
    print("\nSample paper:")
    print(f"  ID: {sample['id']}")
    print(f"  Title: {sample['title'][:80]}...")
    print(f"  Categories: {sample['categories']}")
    print(f"  Date: {sample['update_date']}")
