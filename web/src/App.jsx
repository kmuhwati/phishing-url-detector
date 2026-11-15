import { useEffect, useMemo, useState } from "react";
import { extractFeatures, FEATURE_LABELS, FEATURE_NAMES, parseUrl, registeredDomain } from "./features.js";
import { explain } from "./model.js";

const REPO = "https://github.com/kmuhwati/phishing-url-detector";
const AUTHOR = "https://kmuhwati.github.io/";
const EXAMPLES = [
  { url: "http://paypal.com.secure-login-verify.xyz/webscr/login.php?cmd=_login", kind: "phish" },
  { url: "https://appleid.apple.com.verify-account.top/", kind: "phish" },
  { url: "https://metamask-wallet-restore.web.app/recover", kind: "phish" },
  { url: "https://en.wikipedia.org/wiki/Phishing", kind: "safe" },
  { url: "https://login.microsoftonline.com/common/oauth2/authorize", kind: "safe" },
  { url: "https://github.com/kmuhwati/mobile-money-fraud-detection", kind: "safe" },
];
// Test-set sizes from reports/metrics.json; the results themselves come from model.json
const TESTS = [
  { name: "Live phishing links", source: "OpenPhish feed, 15 Nov 2026", n: 300, measure: "caught", key: "openphish_recall", good: "high" },
  { name: "Real web pages", source: "Common Crawl, sites never seen in training", n: 9668, measure: "wrongly flagged", key: "commoncrawl_fpr", good: "low" },
  { name: "Popular homepages", source: "Tranco top sites, never seen in training", n: 10000, measure: "wrongly flagged", key: "tranco_fpr", good: "low" },
  { name: "Same dataset, new sites", source: "PhiUSIIL test split, grouped by site", n: 53548, measure: "ROC AUC", key: "in_dataset_auc", good: "high", raw: true },
];
const FLAGS = new Set(["is_ip", "has_port", "punycode", "risky_tld", "tld_com", "tld_org_net", "tld_edu_gov", "free_hosting",
  "url_shortener", "brand_in_host_not_owner", "brand_in_path", "has_at_sign", "double_slash_in_path", "ext_php", "ext_html", "ext_executable"]);
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const show = (name, v) => (FLAGS.has(name) ? (v === 1 ? "yes" : "no") : Number.isInteger(v) ? v : v.toFixed(2));

/** Split a URL into labelled parts for display: what the model ignores, the subdomains, the owner's domain, the path. */
function breakdown(input) {
  const raw = String(input).trim();
  const { host, rest } = parseUrl(raw);
  const scheme = (raw.match(/^[a-z][a-z0-9+.-]*:\/\//i) || [""])[0];
  const www = (raw.slice(scheme.length).match(/^www\d*\./i) || [""])[0];
  const labels = host.split(".").filter(Boolean);
  const isIp = /^\d{1,3}(\.\d{1,3}){3}$/.test(host);
  const reg = isIp ? labels : registeredDomain(labels);
  const sub = labels.slice(0, labels.length - reg.length).join(".");
  const q = rest.search(/[?#]/);
  return {
    scheme, www, sub, domain: reg.join("."), path: q === -1 ? rest : rest.slice(0, q), query: q === -1 ? "" : rest.slice(q),
    subCount: Math.max(0, labels.length - reg.length), tld: isIp ? "IP address" : `.${labels[labels.length - 1] || ""}`,
  };
}

export default function App() {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [input, setInput] = useState(EXAMPLES[0].url);
  const [url, setUrl] = useState(EXAMPLES[0].url);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}model.json`).then((r) => r.json()).then(setModel).catch(setError);
  }, []);

  const result = useMemo(() => {
    if (!model || !url.trim()) return null;
    const features = extractFeatures(url);
    const { probability, contributions } = explain(model, features);
    const factors = FEATURE_NAMES.map((name, i) => ({ name, value: features[i], effect: contributions[i] }))
      .filter((r) => Math.abs(r.effect) > 0.02)
      .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect))
      .slice(0, 7);
    const maxEffect = Math.max(0.01, ...factors.map((f) => Math.abs(f.effect)));
    return { probability, factors, maxEffect, flagged: probability >= model.threshold, parts: breakdown(url) };
  }, [model, url]);

  const submit = (e) => { e.preventDefault(); setUrl(input); };
  const tryExample = (u) => { setInput(u); setUrl(u); };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <span className="name">Link risk review</span>
          <span className="crumb">URL model · runs in your browser</span>
        </div>
        <nav>
          <a href={REPO}>Method and code</a>
          <a href={AUTHOR}>Kumbirai Muhwati</a>
        </nav>
      </header>

      <main className="page">
        <h1>Check a link before you click it</h1>
        <p className="intro">
          A gradient-boosted model scores the address from its text alone and shows which parts drove the score.
          Nothing you type leaves this page.
        </p>

        <form className="search" onSubmit={submit}>
          <label htmlFor="url" className="search-label">URL</label>
          <input id="url" type="text" inputMode="url" autoComplete="off" spellCheck="false" value={input}
                 onChange={(e) => setInput(e.target.value)} placeholder="https://example.com/login" />
          <button type="submit">Review</button>
        </form>
        <div className="examples">
          <span className="ex-label">Examples</span>
          {EXAMPLES.map((ex) => (
            <button key={ex.url} type="button" className={`ex ${ex.kind}`} onClick={() => tryExample(ex.url)} title={ex.url}>
              <i aria-hidden="true" />{ex.url.replace(/^https?:\/\//, "").split(/[/?]/)[0]}
            </button>
          ))}
        </div>

        {error && <p className="notice">Could not load the model: {String(error)}</p>}
        {!model && !error && <p className="notice">Loading the model…</p>}

        {model && result && (
          <section className={`review ${result.flagged ? "flagged" : "clear"}`} aria-live="polite">
            <div className="review-head">
              <h2>Risk evaluation</h2>
              <span className="status">{result.flagged ? "Likely phishing" : "No strong phishing signals"}</span>
            </div>

            <div className="review-body">
              <div className="score-block">
                <span className="k">Risk score</span>
                <span className="score">{Math.round(99 * result.probability)}<small>/ 99</small></span>
                <Scale p={result.probability} threshold={model.threshold} />
                <p className="fine">
                  Alerts at {Math.round(99 * model.threshold)}, the level that flags about 1% of legitimate links in validation.
                </p>
              </div>

              <div className="url-block">
                <span className="k">How the model reads this address</span>
                <p className="url-parts">
                  {result.parts.scheme && <span className="p-ignored" title="Ignored by the model">{result.parts.scheme}</span>}
                  {result.parts.www && <span className="p-ignored" title="Ignored by the model">{result.parts.www}</span>}
                  {result.parts.sub && <span className="p-sub">{result.parts.sub}.</span>}
                  <span className="p-domain">{result.parts.domain}</span>
                  {result.parts.path && <span className="p-path">{result.parts.path}</span>}
                  {result.parts.query && <span className="p-query">{result.parts.query}</span>}
                </p>
                <dl className="kv">
                  <div><dt><i className="dot sub" />Subdomains</dt><dd>{result.parts.subCount}</dd></div>
                  <div><dt><i className="dot domain" />Registered domain</dt><dd>{result.parts.domain || "none"}</dd></div>
                  <div><dt><i className="dot path" />Path and query</dt><dd>{(result.parts.path + result.parts.query).length} chars</dd></div>
                  <div><dt><i className="dot ignored" />Scheme and www.</dt><dd>ignored</dd></div>
                </dl>
              </div>
            </div>

            <div className="factors">
              <h3>What moved the score</h3>
              {result.factors.length === 0 ? (
                <p className="fine">Nothing in this address moved the score much either way.</p>
              ) : (
                <table>
                  <thead>
                    <tr><th>Factor</th><th className="num">Value</th><th className="effect-h"><span>lowers risk</span><span>raises risk</span></th></tr>
                  </thead>
                  <tbody>
                    {result.factors.map((f) => (
                      <tr key={f.name}>
                        <td>{FEATURE_LABELS[f.name]}</td>
                        <td className="num">{show(f.name, f.value)}</td>
                        <td className="effect">
                          <span className="axis" />
                          <span className={`bar ${f.effect > 0 ? "up" : "down"}`} style={{ "--w": `${(50 * Math.abs(f.effect)) / result.maxEffect}%` }}
                                aria-label={f.effect > 0 ? "raises risk" : "lowers risk"} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>
        )}

        {model && (
          <section className="perf">
            <h2>How far to trust it</h2>
            <p className="intro small">The first three test sets come from sources the model never trained on, scored at the same alert level.</p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Test set</th><th className="num">Links</th><th className="num">Result</th></tr>
                </thead>
                <tbody>
                  {TESTS.map((t) => (
                    <tr key={t.key}>
                      <td><span className="t-name">{t.name}</span><span className="t-src">{t.source}</span></td>
                      <td className="num">{t.n.toLocaleString("en-US")}</td>
                      <td className="num">
                        <span className="strong">{t.raw ? model.meta[t.key].toFixed(2) : pct(model.meta[t.key])}</span>
                        <span className="t-src">{t.measure}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="callout">
              <h3>Why not 99.9%?</h3>
              <p>
                In the public dataset most tutorials use, every legitimate URL is a bare <code>https://www.</code> homepage.
                A model trained the usual way learns that rule, scores 0.999 on the dataset, and then flags 99.9% of real
                web pages. This model never sees the scheme or the <code>www.</code>, learns from real page addresses too,
                and is tested on outside sources. It reads only the link text, so treat it as a second opinion.
              </p>
            </div>
          </section>
        )}
      </main>

      <footer className="foot">
        Data: PhiUSIIL (UCI, 2024), Common Crawl CC-MAIN-2026-39, OpenPhish, Tranco. Method, evaluation and tests on <a href={REPO}>GitHub</a>.
      </footer>
    </div>
  );
}

function Scale({ p, threshold }) {
  const n = 25;
  const filled = Math.round(p * n);
  const cut = Math.round(threshold * n);
  return (
    <div className="scale" role="img" aria-label={`Score ${Math.round(99 * p)} of 99, alert level ${Math.round(99 * threshold)}`}>
      <div className="segs">
        {Array.from({ length: n }, (_, i) => (
          <span key={i} className={`${i < filled ? "on" : ""} ${i >= cut ? "hot" : ""}`} />
        ))}
      </div>
      <span className="cut" style={{ left: `${(100 * cut) / n}%` }}><em>alert</em></span>
    </div>
  );
}
