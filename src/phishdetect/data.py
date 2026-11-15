"""Datasets and splits.

Training data
  * PhiUSIIL (UCI, 2024): 235,795 URLs. Its label is 1 = legitimate, 0 = phishing; here phish = 1.
    Every legitimate URL is a bare "https://www.<domain>" homepage, which is a shortcut a model can learn.
  * Common Crawl pages (CC-MAIN-2026-39, from project 06): real page URLs with paths, used as extra
    legitimate examples so "has a path" stops meaning "phishing". 12 of the 16 crawl files train.

Held-out tests the model never trains on
  * OpenPhish community feed: live phishing URLs downloaded on the day of the run.
  * Common Crawl pages from the other 4 crawl files, minus any site seen in training.
  * Tranco top-20,000 site homepages, minus any site seen in training.
"""
from __future__ import annotations

import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu", "ne", "or", "go"}


def host_of(url: str) -> str:
    s = url.strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    host = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].split("@")[-1].split(":")[0].lower()
    return host[4:] if host.startswith("www.") else host


def site_of(url: str) -> str:
    """Registered domain (same heuristic as features.js): the unit we group and de-duplicate by."""
    labels = [p for p in host_of(url).split(".") if p]
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def phiusiil() -> pd.DataFrame:
    df = pd.read_csv(glob.glob(str(RAW / "phiusiil" / "*.csv"))[0], usecols=["URL", "label"])
    return pd.DataFrame({"url": df.URL.astype(str), "phish": 1 - df.label, "source": "phiusiil"})


def commoncrawl() -> pd.DataFrame:
    df = pd.read_csv(RAW / "commoncrawl_urls.csv")
    files = sorted(df.wet_file.unique())
    df["split"] = np.where(df.wet_file.isin(files[:12]), "train", "test")
    return pd.DataFrame({"url": df.url, "phish": 0, "source": "commoncrawl", "split": df.split})


def openphish() -> pd.DataFrame:
    urls = [u.strip() for u in (RAW / "openphish_feed.txt").read_text(encoding="utf-8").splitlines() if u.strip()]
    return pd.DataFrame({"url": sorted(set(urls)), "phish": 1, "source": "openphish"})


def tranco(n: int = 20_000) -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "tranco.zip") as z:
        name = [f for f in z.namelist() if f.endswith(".csv")][0]
        domains = pd.read_csv(z.open(name), header=None, names=["rank", "domain"], nrows=n).domain
    return pd.DataFrame({"url": "https://" + domains.astype(str), "phish": 0, "source": "tranco"})
