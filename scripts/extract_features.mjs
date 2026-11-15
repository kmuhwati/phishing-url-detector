// Compute URL features with the web app's own code, for the Python training pipeline.
//   node scripts/extract_features.mjs <urls.txt> <out.csv> [naive]
// Input: one URL per line. Output: CSV with a header row of feature names.
import { createReadStream, createWriteStream } from "node:fs";
import { createInterface } from "node:readline";
import { extractFeatures, extractNaiveFeatures, FEATURE_NAMES, NAIVE_FEATURE_NAMES } from "../web/src/features.js";

const [input, output, mode] = process.argv.slice(2);
const naive = mode === "naive";
const names = naive ? NAIVE_FEATURE_NAMES : FEATURE_NAMES;
const extract = naive ? extractNaiveFeatures : extractFeatures;

const out = createWriteStream(output);
out.write(`${names.join(",")}\n`);
const lines = createInterface({ input: createReadStream(input, "utf8"), crlfDelay: Infinity });
let n = 0;
for await (const line of lines) {
  out.write(`${extract(line).map((v) => (Number.isInteger(v) ? v : v.toFixed(6))).join(",")}\n`);
  n++;
}
out.end(() => console.error(`features for ${n} URLs -> ${output}`));
