"""Train the naive and robust phishing-URL models, evaluate them on held-out sources, export the robust
model for the browser (web/public/model.json) and write the report.

    python scripts/download_data.py
    python scripts/collect_commoncrawl_urls.py ../06-llm-data-refinery/data/output/corpus
    python scripts/run_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishdetect.pipeline import run  # noqa: E402

if __name__ == "__main__":
    m = run()
    print(json.dumps(m["results"], indent=2))
    print(json.dumps({k: m[k] for k in ("datasets", "naive_threshold", "robust_threshold", "naive_top_features", "robust_top_features")}, indent=2))
