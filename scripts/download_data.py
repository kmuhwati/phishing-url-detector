"""Download PhiUSIIL, the Tranco top-1M list and today's OpenPhish feed into data/raw/."""
import io
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UA = {"User-Agent": "kmuhwati-phishing-research/1.0 (https://github.com/kmuhwati)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        return r.read()


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(get("https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"))).extractall(RAW / "phiusiil")
    (RAW / "tranco.zip").write_bytes(get("https://tranco-list.eu/top-1m.csv.zip"))
    (RAW / "openphish_feed.txt").write_bytes(get("https://openphish.com/feed.txt"))
    print("downloaded to", RAW)
