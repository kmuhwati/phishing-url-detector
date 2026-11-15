// The browser must reproduce the Python model's scores (fixtures written by scripts/run_pipeline.py).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { extractFeatures, FEATURE_NAMES, parseUrl } from "../src/features.js";
import { explain } from "../src/model.js";

const model = JSON.parse(readFileSync(new URL("../public/model.json", import.meta.url), "utf8"));
const fixtures = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url), "utf8"));

test("model features match the extractor", () => {
  assert.deepEqual(model.features, FEATURE_NAMES);
});

test("JavaScript scores match LightGBM on held-out URLs", () => {
  let worst = 0;
  for (const f of fixtures) {
    const { raw } = explain(model, extractFeatures(f.url));
    worst = Math.max(worst, Math.abs(raw - f.raw));
  }
  assert.ok(worst < 1e-5, `largest difference ${worst}`);
});

test("contributions add up to the score", () => {
  const { raw, base, contributions } = explain(model, extractFeatures("http://secure-login.paypal.com.verify-account.xyz/webscr.php"));
  const total = base + contributions.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(total - raw) < 1e-9);
});

test("parseUrl strips scheme, www and a bare slash", () => {
  assert.deepEqual(
    [parseUrl("https://www.Example.com/").host, parseUrl("https://www.Example.com/").rest],
    ["example.com", ""],
  );
  assert.equal(parseUrl("example.com/a?b=1").rest, "/a?b=1");
});

test("brand on a site the brand does not own is flagged, the real site is not", () => {
  const i = FEATURE_NAMES.indexOf("brand_in_host_not_owner");
  assert.equal(extractFeatures("https://www.paypal.com")[i], 0);
  assert.equal(extractFeatures("http://paypal.com.account-check.ru/login")[i], 1);
});
