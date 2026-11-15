// Gradient-boosted tree model exported from LightGBM (web/public/model.json), evaluated in the browser.
// Each tree is a list of nodes [feature, threshold, left, right, value]; leaves have feature -1.

const round6 = (v) => Math.round(v * 1e6) / 1e6; // the training pipeline saw features rounded to 6 decimals

/** Raw score (log-odds) and per-feature contributions for one feature vector. */
export function explain(model, rawFeatures) {
  const x = rawFeatures.map(round6);
  const contributions = new Array(model.features.length).fill(0);
  let raw = 0;
  let base = 0;
  for (const tree of model.trees) {
    // Contributions follow the path from root to leaf: each split is credited with the change in the
    // node value it causes (Saabas attribution). Root values sum to the base score.
    let node = 0;
    base += tree[0][4];
    while (tree[node][0] !== -1) {
      const [feature, threshold, left, right, value] = tree[node];
      const next = x[feature] <= threshold ? left : right;
      contributions[feature] += tree[next][4] - value;
      node = next;
    }
    raw += tree[node][4];
  }
  return { raw, base, contributions, probability: 1 / (1 + Math.exp(-raw)) };
}

export function predict(model, rawFeatures) {
  return explain(model, rawFeatures).probability;
}
