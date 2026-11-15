// URL features: the single implementation used by the web app and, through Node, by the Python
// training pipeline (scripts/extract_features.mjs). One source means the model in the browser sees
// exactly what it was trained on. Everything is computed from the URL text alone; no page is fetched.

const RISKY_TLDS = new Set(["xyz", "top", "shop", "online", "site", "live", "click", "info", "icu", "buzz", "cfd",
  "sbs", "lol", "rest", "monster", "cyou", "work", "support", "tk", "ml", "ga", "cf", "gq", "zip", "mov"]);
const FREE_HOSTS = ["firebaseapp.com", "web.app", "ipfs", "weebly.com", "wixsite.com", "blogspot.com",
  "000webhostapp.com", "glitch.me", "netlify.app", "vercel.app", "pages.dev", "workers.dev", "herokuapp.com",
  "repl.co", "duckdns.org", "ngrok", "r2.dev", "square.site", "godaddysites.com", "webflow.io",
  "gitbook.io", "translate.goog", "appspot.com", "azurewebsites.net", "framer.app"];
const SHORTENERS = new Set(["bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rebrand.ly", "ow.ly",
  "s.id", "shorturl.at", "rb.gy", "t.ly", "tiny.cc", "buff.ly"]);
const BRANDS = ["paypal", "apple", "icloud", "microsoft", "office365", "outlook", "amazon", "netflix", "google",
  "gmail", "facebook", "instagram", "whatsapp", "dhl", "usps", "fedex", "chase", "wellsfargo", "bankofamerica",
  "binance", "coinbase", "metamask", "ledger", "adobe", "dropbox", "docusign", "linkedin", "steam", "roblox"];
// Registered domains the brands above really own; a brand name on any other domain is a warning sign
const BRAND_DOMAINS = new Set(["paypal.com", "apple.com", "icloud.com", "microsoft.com", "microsoftonline.com",
  "office.com", "office365.com", "outlook.com", "live.com", "windows.net", "azure.com", "amazon.com", "amazon.co.uk",
  "amazon.de", "amazonaws.com", "netflix.com", "google.com", "gmail.com", "googleapis.com", "googleusercontent.com",
  "youtube.com", "facebook.com", "fb.com", "instagram.com", "whatsapp.com", "whatsapp.net", "dhl.com", "dhl.de",
  "usps.com", "fedex.com", "chase.com", "wellsfargo.com", "bankofamerica.com", "binance.com", "coinbase.com",
  "metamask.io", "ledger.com", "adobe.com", "dropbox.com", "docusign.com", "docusign.net", "linkedin.com",
  "steampowered.com", "steamcommunity.com", "roblox.com"]);
const SUSPICIOUS = ["login", "signin", "sign-in", "verify", "verification", "secure", "account", "update", "bank",
  "confirm", "password", "wallet", "webscr", "support", "billing", "unlock", "suspend", "authenticate", "recover",
  "invoice", "payment", "auth", "session", "token", "validate", "claim", "airdrop", "reward"];
const SECOND_LEVEL = new Set(["co", "com", "org", "net", "gov", "ac", "edu", "ne", "or", "go"]);

export const FEATURE_NAMES = [
  "host_length", "host_labels", "subdomains", "host_hyphens", "host_digits", "host_digit_ratio", "host_entropy",
  "longest_host_label", "is_ip", "has_port", "punycode", "tld_length", "risky_tld", "tld_com", "tld_org_net",
  "tld_edu_gov", "free_hosting", "url_shortener", "brand_in_host_not_owner", "brand_in_path", "host_suspicious_words",
  "path_length", "path_depth", "query_length", "query_params", "path_digits", "path_special_chars", "path_entropy",
  "longest_path_token", "has_at_sign", "double_slash_in_path", "ext_php", "ext_html", "ext_executable",
  "suspicious_words",
];

// Labels shown in the app next to each reason
export const FEATURE_LABELS = {
  host_length: "Hostname length", host_labels: "Parts in the hostname", subdomains: "Number of subdomains",
  host_hyphens: "Hyphens in the hostname", host_digits: "Digits in the hostname", host_digit_ratio: "Share of digits in the hostname",
  host_entropy: "Randomness of the hostname", longest_host_label: "Longest part of the hostname", is_ip: "Raw IP address instead of a name",
  has_port: "Explicit port number", punycode: "Internationalised (punycode) hostname", tld_length: "Length of the domain ending",
  risky_tld: "Domain ending often abused for phishing", tld_com: ".com domain", tld_org_net: ".org or .net domain",
  tld_edu_gov: ".edu or .gov domain", free_hosting: "Free hosting or site-builder platform", url_shortener: "Link shortener",
  brand_in_host_not_owner: "Brand name on a domain the brand does not own", brand_in_path: "Brand name in the path",
  host_suspicious_words: "Words like 'login' or 'verify' in the hostname", path_length: "Path length", path_depth: "Path depth",
  query_length: "Query string length", query_params: "Query parameters", path_digits: "Digits in the path",
  path_special_chars: "Special characters in the path", path_entropy: "Randomness of the path", longest_path_token: "Longest token in the path",
  has_at_sign: "@ sign in the URL", double_slash_in_path: "Double slash inside the path", ext_php: "Page ends in .php",
  ext_html: "Page ends in .html", ext_executable: "Link to an executable or archive", suspicious_words: "Words like 'login', 'verify' or 'wallet'",
};

function entropy(s) {
  if (!s) return 0;
  const counts = new Map();
  for (const ch of s) counts.set(ch, (counts.get(ch) || 0) + 1);
  let h = 0;
  for (const n of counts.values()) {
    const p = n / s.length;
    h -= p * Math.log2(p);
  }
  return h;
}

const countMatches = (s, re) => (s.match(re) || []).length;

/** Split a URL into lower-case hostname (without a leading "www.") and everything after it. */
export function parseUrl(input) {
  let s = String(input).trim();
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) s = `http://${s}`;
  let host = "";
  let rest = "";
  try {
    const u = new URL(s);
    host = u.hostname.toLowerCase();
    rest = `${u.pathname}${u.search}${u.hash}`;
  } catch {
    const body = s.replace(/^[a-z][a-z0-9+.-]*:\/\//i, "");
    const cut = body.search(/[/?#]/);
    host = (cut === -1 ? body : body.slice(0, cut)).toLowerCase();
    rest = cut === -1 ? "" : body.slice(cut);
  }
  host = host.replace(/^www\d*\./, "").replace(/\.$/, "");
  if (rest === "/") rest = "";
  return { host, rest, hasPort: /^[a-z][a-z0-9+.-]*:\/\/[^/?#]*:\d+/i.test(s), raw: String(input) };
}

export function registeredDomain(labels) {
  if (labels.length >= 3 && labels[labels.length - 1].length === 2 && SECOND_LEVEL.has(labels[labels.length - 2])) {
    return labels.slice(-3);
  }
  return labels.slice(-2);
}

/** Feature vector (same order as FEATURE_NAMES) for one URL. */
export function extractFeatures(input) {
  const { host, rest, hasPort, raw } = parseUrl(input);
  const isIp = /^\d{1,3}(\.\d{1,3}){3}$/.test(host) || host.startsWith("[");
  const labels = host.split(".").filter(Boolean);
  const reg = isIp ? labels : registeredDomain(labels);
  const tld = isIp ? "" : labels[labels.length - 1] || "";
  const hostDigits = countMatches(host, /\d/g);
  const lowerRest = rest.toLowerCase();
  const brandInHost = !BRAND_DOMAINS.has(reg.join(".")) && BRANDS.some((b) => host.includes(b));
  const pathOnly = rest.split(/[?#]/)[0];
  const query = rest.includes("?") ? rest.slice(rest.indexOf("?") + 1).split("#")[0] : "";
  const tokens = rest.split(/[/?=&._\-#]+/).filter(Boolean);
  const words = (text) => SUSPICIOUS.reduce((n, w) => n + (text.includes(w) ? 1 : 0), 0);
  return [
    host.length,
    labels.length,
    Math.max(0, labels.length - reg.length),
    countMatches(host, /-/g),
    hostDigits,
    host.length ? hostDigits / host.length : 0,
    entropy(host),
    labels.reduce((m, l) => Math.max(m, l.length), 0),
    isIp ? 1 : 0,
    hasPort ? 1 : 0,
    host.includes("xn--") ? 1 : 0,
    tld.length,
    RISKY_TLDS.has(tld) ? 1 : 0,
    tld === "com" ? 1 : 0,
    tld === "org" || tld === "net" ? 1 : 0,
    tld === "edu" || tld === "gov" ? 1 : 0,
    FREE_HOSTS.some((f) => host === f || host.endsWith(`.${f}`) || host.includes(f)) ? 1 : 0,
    SHORTENERS.has(host) ? 1 : 0,
    brandInHost ? 1 : 0,
    BRANDS.some((b) => lowerRest.includes(b)) ? 1 : 0,
    words(host),
    rest.length,
    pathOnly.split("/").filter(Boolean).length,
    query.length,
    query ? query.split("&").length : 0,
    countMatches(rest, /\d/g),
    countMatches(rest, /[%~_=@&!$]/g),
    entropy(rest),
    tokens.reduce((m, t) => Math.max(m, t.length), 0),
    raw.includes("@") ? 1 : 0,
    rest.slice(1).includes("//") ? 1 : 0,
    /\.php$/i.test(pathOnly) ? 1 : 0,
    /\.html?$/i.test(pathOnly) ? 1 : 0,
    /\.(exe|zip|apk|scr|msi|rar|7z)$/i.test(pathOnly) ? 1 : 0,
    words(`${host} ${lowerRest}`),
  ];
}

/**
 * The naive model's features: the same vector computed on the URL exactly as given, plus whether
 * it starts with "https://" and "www.". Used only to measure the dataset shortcut, never served.
 */
export const NAIVE_FEATURE_NAMES = ["starts_https", "has_www", "raw_length", "has_path", ...FEATURE_NAMES];
export function extractNaiveFeatures(input) {
  const s = String(input).trim();
  const { rest } = parseUrl(s);
  return [/^https:\/\//i.test(s) ? 1 : 0, /^([a-z]+:\/\/)?www\d*\./i.test(s) ? 1 : 0, s.length, rest ? 1 : 0, ...extractFeatures(s)];
}
