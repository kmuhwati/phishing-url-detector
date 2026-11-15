"""Tests for the data helpers, the threshold rule and the Node feature bridge (no download needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishdetect import data, pipeline  # noqa: E402


def test_site_of_groups_by_registered_domain():
    assert data.site_of("https://www.example.com/a") == "example.com"
    assert data.site_of("http://login.secure.example.co.uk/x") == "example.co.uk"
    assert data.site_of("caseid123.firebaseapp.com") == "firebaseapp.com"
    assert data.host_of("https://user@Mail.Example.com:8443/inbox") == "mail.example.com"


def test_threshold_flags_the_target_share_of_legitimate_urls():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000), np.ones(1_000)]
    p = np.r_[rng.random(10_000) * 0.8, 0.5 + rng.random(1_000) * 0.5]
    thr = pipeline.threshold_at_fpr(y, p, 0.01)
    assert abs((p[y == 0] >= thr).mean() - 0.01) < 0.002


def test_scores_reports_recall_and_false_positive_rate():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    s = pipeline.scores(y, p, 0.5)
    assert s["recall"] == 0.5 and s["false_positive_rate"] == 0.5 and s["precision"] == 0.5


def test_node_features_match_the_browser_feature_list():
    urls = pd.Series(["https://www.paypal.com", "http://paypal.com.verify-login.xyz/webscr.php?x=1", "bit.ly/abc"])
    robust = pipeline.features(urls)
    naive = pipeline.features(urls, naive=True)
    assert robust.shape == (3, 35) and naive.shape == (3, 39)
    assert not robust.isna().any().any()
    assert robust.loc[0, "brand_in_host_not_owner"] == 0 and robust.loc[1, "brand_in_host_not_owner"] == 1
    assert naive.loc[0, "has_www"] == 1 and naive.loc[0, "starts_https"] == 1
    # "www." and the scheme do not reach the robust model at all
    assert robust.loc[0].tolist() == pipeline.features(pd.Series(["paypal.com"])).loc[0].tolist()
