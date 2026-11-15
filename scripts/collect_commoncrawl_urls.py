"""Copy the page URLs (and which crawl file each came from) out of project 06's refined corpus.

Project 06 (llm-pretraining-data-refinery) processed 16 WET files of Common Crawl CC-MAIN-2026-39.
Its corpus keeps each page's URL and a doc_id that starts with the source file's key.

    python scripts/collect_commoncrawl_urls.py ../06-llm-data-refinery/data/output/corpus
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "commoncrawl_urls.csv"

if __name__ == "__main__":
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "../06-llm-data-refinery/data/output/corpus")
    df = pd.read_parquet(corpus, columns=["doc_id", "url"])
    df["wet_file"] = df.doc_id.str.rsplit("-", n=1).str[0]
    df[["url", "wet_file"]].drop_duplicates("url").to_csv(OUT, index=False)
    print(f"{len(df):,} URLs from {df.wet_file.nunique()} crawl files -> {OUT}")
