"""Train the naive and robust URL models, evaluate them honestly, and export the robust one for the browser."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from . import data, plotstyle

ROOT = data.ROOT
REPORTS, FIGURES, TABLES = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "reports" / "tables"
WEB_PUBLIC, WEB_TEST = ROOT / "web" / "public", ROOT / "web" / "test"
SEED = 7
TARGET_FPR = 0.01            # the alert threshold is set on validation data to flag ~1% of legitimate URLs
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=40,
              colsample_bytree=0.9, subsample=0.9, subsample_freq=1, reg_lambda=1.0, random_state=SEED,
              n_jobs=os.cpu_count(), verbose=-1)


def features(urls: pd.Series, naive: bool = False) -> pd.DataFrame:
    """Run web/src/features.js through Node, so training sees exactly what the browser computes."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "urls.txt", Path(tmp) / "features.csv"
        src.write_text("\n".join(urls.str.replace(r"[\r\n]", "", regex=True)), encoding="utf-8")
        cmd = ["node", str(ROOT / "scripts" / "extract_features.mjs"), str(src), str(out)] + (["naive"] if naive else [])
        subprocess.run(cmd, check=True, capture_output=True)
        return pd.read_csv(out)


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float) -> float:
    legit = np.sort(p[y == 0])
    return float(legit[int(np.ceil((1 - fpr) * len(legit))) - 1]) if len(legit) else 0.5


def scores(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    flag = p >= thr
    out = {"n": int(len(y)), "phish": int(y.sum())}
    if y.min() != y.max():
        out |= {"roc_auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p))}
    if y.sum():
        out["recall"] = float(flag[y == 1].mean())
    if (y == 0).sum():
        out["false_positive_rate"] = float(flag[y == 0].mean())
    if flag.sum():
        out["precision"] = float(y[flag].mean())
    return out


def run() -> dict:
    plotstyle.apply()
    for d in (FIGURES, TABLES, WEB_PUBLIC, WEB_TEST):
        d.mkdir(parents=True, exist_ok=True)

    phi, cc, fresh_phish, top_sites = data.phiusiil(), data.commoncrawl(), data.openphish(), data.tranco()
    phi["site"], cc["site"] = phi.url.map(data.site_of), cc.url.map(data.site_of)
    shortcut = phi.assign(has_path=phi.url.map(lambda u: bool(data.host_of(u)) and "/" in u.split("://", 1)[-1].rstrip("/")),
                          https=phi.url.str.startswith("https://"), www=phi.url.str.contains("://www.", regex=False))
    shortcut_table = shortcut.groupby("phish")[["has_path", "https", "www"]].mean()
    shortcut_table.to_csv(TABLES / "phiusiil_shortcut.csv")

    # ---------------- naive model: PhiUSIIL as given, random split (what most tutorials do)
    xn = features(phi.url, naive=True)
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(len(phi)) < 0.2
    naive = lgb.LGBMClassifier(**PARAMS).fit(xn[~test_mask], phi.phish[~test_mask])
    naive_thr = threshold_at_fpr(phi.phish[~test_mask].to_numpy(), naive.predict_proba(xn[~test_mask])[:, 1], TARGET_FPR)

    # ---------------- robust model: no scheme/www, Common Crawl pages as extra legitimate URLs, grouped by site
    train_pool = pd.concat([phi.assign(split="train"), cc[cc.split == "train"]], ignore_index=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit_idx, test_idx = next(gss.split(train_pool, groups=train_pool.site))
    fit_part, in_test = train_pool.iloc[fit_idx], train_pool.iloc[test_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    tr_idx, val_idx = next(gss_val.split(fit_part, groups=fit_part.site))
    train, val = fit_part.iloc[tr_idx], fit_part.iloc[val_idx]
    xr_train, xr_val, xr_in = features(train.url), features(val.url), features(in_test.url)
    robust = lgb.LGBMClassifier(**PARAMS).fit(xr_train, train.phish)
    robust_thr = threshold_at_fpr(val.phish.to_numpy(), robust.predict_proba(xr_val)[:, 1], TARGET_FPR)

    # ---------------- held-out sources, minus any site the robust model trained on
    seen = set(train_pool.site)
    cc_test = cc[(cc.split == "test") & ~cc.site.isin(seen)]
    top_sites = top_sites[~top_sites.url.map(data.site_of).isin(seen)].head(10_000)
    held_out = {"openphish_live": fresh_phish, "commoncrawl_heldout": cc_test, "tranco_top_sites": top_sites}

    results = {"naive": {}, "robust": {}}
    results["naive"]["in_dataset_random_split"] = scores(
        phi.phish[test_mask].to_numpy(), naive.predict_proba(xn[test_mask])[:, 1], naive_thr)
    results["robust"]["in_dataset_grouped_split"] = scores(
        in_test.phish.to_numpy(), robust.predict_proba(xr_in)[:, 1], robust_thr)
    for name, df in held_out.items():
        results["naive"][name] = scores(df.phish.to_numpy(), naive.predict_proba(features(df.url, naive=True))[:, 1], naive_thr)
        results["robust"][name] = scores(df.phish.to_numpy(), robust.predict_proba(features(df.url))[:, 1], robust_thr)

    importance = pd.Series(robust.booster_.feature_importance("gain"), index=xr_train.columns).sort_values(ascending=False)
    importance = importance / importance.sum()
    importance.to_csv(TABLES / "robust_feature_importance.csv", header=["gain_share"])
    naive_imp = pd.Series(naive.booster_.feature_importance("gain"), index=xn.columns).sort_values(ascending=False)
    naive_imp = naive_imp / naive_imp.sum()

    metrics = {
        "datasets": {"phiusiil_urls": int(len(phi)), "phiusiil_phish": int(phi.phish.sum()),
                     "commoncrawl_train_urls": int((cc.split == "train").sum()),
                     "openphish_live_urls": int(len(fresh_phish)), "commoncrawl_heldout_urls": int(len(cc_test)),
                     "tranco_heldout_sites": int(len(top_sites)),
                     "train_urls": int(len(train)), "val_urls": int(len(val)), "in_dataset_test_urls": int(len(in_test))},
        "phiusiil_shortcut": {("phishing" if k else "legitimate"): {c: float(v) for c, v in row.items()}
                              for k, row in shortcut_table.iterrows()},
        "target_fpr": TARGET_FPR, "naive_threshold": naive_thr, "robust_threshold": robust_thr,
        "results": results,
        "naive_top_features": naive_imp.head(5).round(4).to_dict(),
        "robust_top_features": importance.head(8).round(4).to_dict(),
        "model": {"trees": PARAMS["n_estimators"], "leaves_per_tree": PARAMS["num_leaves"], "features": int(xr_train.shape[1])},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    export_model(robust, robust_thr, list(xr_train.columns), metrics)
    write_fixtures(robust, in_test)
    figures(metrics, importance)
    results_md(metrics)
    return metrics


def export_model(model: lgb.LGBMClassifier, threshold: float, names: list[str], metrics: dict) -> None:
    """Compact tree arrays for web/src/model.js: each node is [feature, threshold, left, right, value]."""
    dump = model.booster_.dump_model()
    trees = []
    for t in dump["tree_info"]:
        nodes = []

        def walk(node):
            i = len(nodes)
            nodes.append(None)
            if "leaf_value" in node:
                nodes[i] = [-1, 0, -1, -1, round(node["leaf_value"], 7)]
                return i
            assert node["decision_type"] == "<=", node["decision_type"]
            left, right = walk(node["left_child"]), walk(node["right_child"])
            nodes[i] = [node["split_feature"], node["threshold"], left, right, round(node["internal_value"], 7)]
            return i

        walk(t["tree_structure"])
        trees.append(nodes)
    held = metrics["results"]["robust"]
    payload = {"features": names, "threshold": threshold, "trees": trees,
               "meta": {"trained_on": "PhiUSIIL + Common Crawl CC-MAIN-2026-39 pages",
                        "in_dataset_auc": held["in_dataset_grouped_split"]["roc_auc"],
                        "openphish_recall": held["openphish_live"]["recall"],
                        "commoncrawl_fpr": held["commoncrawl_heldout"]["false_positive_rate"],
                        "tranco_fpr": held["tranco_top_sites"]["false_positive_rate"]}}
    (WEB_PUBLIC / "model.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def write_fixtures(model: lgb.LGBMClassifier, in_test: pd.DataFrame) -> None:
    """URLs with the Python model's raw scores, for the JavaScript predictor's parity test.
    Only published-dataset URLs are used; the live OpenPhish feed is not redistributed."""
    sample = in_test.sample(200, random_state=SEED).reset_index(drop=True)
    raw = model.booster_.predict(features(sample.url), raw_score=True)
    benign = ["https://www.wikipedia.org", "https://github.com/kmuhwati", "https://kmuhwati.github.io/"]
    raw_benign = model.booster_.predict(features(pd.Series(benign)), raw_score=True)
    cases = [{"url": u, "raw": float(r)} for u, r in zip(list(sample.url) + benign, list(raw) + list(raw_benign))]
    (WEB_TEST / "fixtures.json").write_text(json.dumps(cases), encoding="utf-8")


def figures(metrics: dict, importance: pd.Series) -> None:
    js = "import('./web/src/features.js').then(m => process.stdout.write(JSON.stringify(m.FEATURE_LABELS)))"
    from_js = json.loads(subprocess.run(["node", "-e", js], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    r = metrics["results"]
    groups = [("Same dataset\n(test split)", "in_dataset"), ("Live phishing\n(OpenPhish)", "openphish_live"),
              ("Real web pages\n(Common Crawl)", "commoncrawl_heldout"), ("Top sites\n(Tranco)", "tranco_top_sites")]

    def get(model, key):
        k = next(x for x in r[model] if x.startswith(key))
        s = r[model][k]
        return s.get("roc_auc") if key == "in_dataset" else (s.get("recall") if key == "openphish_live" else s["false_positive_rate"])

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))
    titles = ["ROC AUC ↑", "Phishing caught ↑", "Real pages flagged ↓", "Top sites flagged ↓"]
    for ax, (label, key), title in zip(axes, groups, titles):
        vals = [get("naive", key), get("robust", key)]
        bars = ax.bar(["Naive", "Robust"], vals, color=[plotstyle.NEUTRAL, plotstyle.SERIES[0]], width=0.6)
        pct_label = lambda v: f"{100 * v:.2f}%" if 0.9995 <= v < 1 else f"{100 * v:.1f}%"  # noqa: E731  never round 99.97% up to 100%
        ax.bar_label(bars, labels=[f"{v:.3f}" if key == "in_dataset" else pct_label(v) for v in vals], padding=2, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(label.replace("\n", " "), fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
    fig.suptitle("A 0.999 score on the dataset is not a working detector", x=0.02, ha="left", fontsize=12, weight="bold")
    plotstyle.save(fig, FIGURES / "01_naive_vs_robust.png")

    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh([from_js.get(n, n) for n in top.index], 100 * top.values, color=plotstyle.SERIES[0], height=0.62)
    ax.set_xlabel("Share of split gain (%)")
    ax.set_title("What the robust model relies on")
    plotstyle.save(fig, FIGURES / "02_feature_importance.png")

    sc = metrics["phiusiil_shortcut"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(3)
    for i, cls in enumerate(["legitimate", "phishing"]):
        vals = [sc[cls][c] for c in ("has_path", "https", "www")]
        bars = ax.bar(x + (i - 0.5) * 0.36, [100 * v for v in vals], 0.36, label=cls,
                      color=[plotstyle.SERIES[2], plotstyle.SERIES[1]][i])
        ax.bar_label(bars, labels=[f"{100 * v:.0f}%" for v in vals], padding=2, fontsize=9)
    ax.set_xticks(x, ["has a path", "starts with https://", "starts with www."])
    ax.set_ylabel("Share of URLs (%)")
    ax.set_ylim(0, 118)
    ax.set_title("The shortcut in PhiUSIIL: every legitimate URL is a bare https://www. homepage")
    ax.legend()
    plotstyle.save(fig, FIGURES / "03_dataset_shortcut.png")


def results_md(metrics: dict) -> None:
    lines = ["# Results (auto-generated by scripts/run_pipeline.py)\n"]
    for model in ("naive", "robust"):
        rows = [{"test set": k, **v} for k, v in metrics["results"][model].items()]
        lines += [f"\n## {model.capitalize()} model\n", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")]
    lines.append(f"\n## All metrics\n\n```json\n{json.dumps(metrics, indent=2)}\n```\n")
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
