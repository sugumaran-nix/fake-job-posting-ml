"""
analyzer.py  —  Automated Company URL Fraud Detection Module
=============================================================
Evaluates company website URLs using:
  1. HTTPS usage
  2. Suspicious TLD detection
  3. Free hosting platform detection
  4. URL entropy (randomness score)
  5. Suspicious keyword detection
  6. Subdomain depth analysis
  7. Typosquatting / brand impersonation
  8. IP address instead of domain
  9. Domain digit density
 10. Domain length anomaly

Also performs company name credibility analysis.
No phone/email — replaced by automated URL analysis.

Fake Job Posting Prediction
"""

import re
import math
from urllib.parse import urlparse
from collections import Counter


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _risk(score: int) -> str:
    if score >= 55: return "high"
    if score >= 22: return "medium"
    return "low"

def _entropy(s: str) -> float:
    """Shannon entropy of a string — high entropy = random/generated."""
    if not s: return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

FREE_HOSTING = [
    "000webhost", "wix.com", "weebly.com", "blogspot", "wordpress.com",
    "sites.google", "github.io", "netlify.app", "vercel.app", "glitch.me",
    "repl.co", "web.app", "firebaseapp", "x10host", "byethost", "heliohost",
    "hostfree", "epizy.com", "rf.gd", "esy.es", "atwebpages", "bravenet",
    "000.pe", "freehosting", "infinityfree", "awardspace"
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".xyz", ".top", ".club", ".online", ".site",
    ".biz", ".ws", ".cc", ".click", ".loan",
    ".win", ".bid", ".trade", ".link", ".info"
]

TRUSTED_TLDS = [
    ".com", ".in", ".co.in", ".org", ".net",
    ".gov.in", ".edu", ".ac.in", ".io", ".co"
]

SCAM_URL_KEYWORDS = [
    "earn", "quickmoney", "fastcash", "guaranteed", "freejob",
    "workfromhome", "wfhjob", "job4u", "job2u", "easyjob",
    "dailyincome", "passiveincome", "homejob", "instantjob",
    "getrich", "richfast", "moneynow", "jobnow", "urgent-hiring",
    "registration-fee", "joining-fee"
]

BRAND_DOMAINS = {
    "infosys":   "infosys.com",
    "tcs":       "tcs.com",
    "wipro":     "wipro.com",
    "amazon":    "amazon.in",
    "flipkart":  "flipkart.com",
    "google":    "google.com",
    "microsoft": "microsoft.com",
    "hdfc":      "hdfcbank.com",
    "icici":     "icicibank.com",
    "paytm":     "paytm.com",
    "cognizant": "cognizant.com",
    "accenture": "accenture.com",
    "hcl":       "hcltech.com",
}

KNOWN_LEGIT_COS = {
    "infosys", "tcs", "tata consultancy", "wipro", "hcl", "tech mahindra",
    "accenture", "cognizant", "capgemini", "ibm", "oracle", "amazon",
    "google", "microsoft", "apple", "meta", "samsung", "reliance", "tata",
    "bajaj", "hdfc", "icici", "sbi", "axis bank", "flipkart", "swiggy",
    "zomato", "ola", "paytm", "mphasis", "hexaware", "bosch", "larsen"
}

SCAM_COMPANY_KW = [
    "earn", "quick earn", "instant earn", "guaranteed income",
    "global earn", "home earn", "easy money", "fast cash", "work easy",
    "simple work", "network marketing", "mlm", "multi level", "pyramid",
    "passive income", "get rich", "daily earning", "zero investment"
]


# ══════════════════════════════════════════════════════════════
#  URL FRAUD DETECTION  (main module)
# ══════════════════════════════════════════════════════════════

def analyse_url(url: str) -> dict:
    """
    Automated company URL fraud detection.
    Evaluates: HTTPS, TLD, free hosting, entropy, keywords,
               subdomain depth, typosquatting, IP, digits, length.
    Returns score (0–100), risk_level, label, flags, techniques used.
    """
    if not url or not url.strip():
        return {
            "score": 35,
            "label": "Unverifiable",
            "risk_level": "medium",
            "summary": "⚠️  No website URL — cannot verify company legitimacy",
            "flags": [("🟡", "No company website provided",
                       "Every legitimate company has an official website for recruitment")],
            "techniques": [],
            "domain": "",
            "raw": ""
        }

    raw = url.strip()
    u   = raw.lower()
    if not u.startswith("http"):
        u = "https://" + u

    try:
        parsed = urlparse(u)
        full_domain = (parsed.netloc or parsed.path.split("/")[0]).replace("www.", "")
        domain_name = full_domain.split(".")[0] if "." in full_domain else full_domain
    except Exception:
        full_domain = domain_name = u

    flags      = []
    score      = 0
    techniques = []

    # ── 1. HTTPS USAGE ────────────────────────────────────────
    techniques.append("HTTPS Validation")
    if raw.strip().startswith("http://"):
        flags.append(("🔴", "HTTP used — no SSL/TLS encryption",
                       "All legitimate business websites enforce HTTPS. HTTP = untrusted."))
        score += 25
    else:
        flags.append(("✅", "HTTPS enabled — SSL/TLS present",
                       "Secure connection is a positive credibility indicator"))

    # ── 2. IP ADDRESS INSTEAD OF DOMAIN ───────────────────────
    techniques.append("IP vs Domain Check")
    if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", u):
        flags.append(("🔴", "Raw IP address used instead of domain",
                       "Legitimate companies never use raw IP addresses for their websites"))
        score += 80
        risk = _risk(min(score, 100))
        return _build_result(score, flags, techniques, full_domain, raw, risk)

    # ── 3. FREE HOSTING PLATFORM ──────────────────────────────
    techniques.append("Free Hosting Detection")
    for fh in FREE_HOSTING:
        if fh in u:
            flags.append(("🔴", f"Free hosting detected: '{fh}'",
                           "No legitimate company uses free website hosting for official business"))
            score += 60
            break

    # ── 4. SUSPICIOUS TLD ─────────────────────────────────────
    techniques.append("TLD Risk Scoring")
    tld_flagged = False
    for tld in SUSPICIOUS_TLDS:
        if full_domain.endswith(tld):
            flags.append(("🔴", f"High-risk free TLD: '{tld}'",
                           f"TLDs like '{tld}' are free/cheap and heavily exploited in fraud"))
            score += 50
            tld_flagged = True
            break
    if not tld_flagged:
        for tld in TRUSTED_TLDS:
            if full_domain.endswith(tld):
                flags.append(("✅", f"Trusted TLD: '{tld}'",
                               "Standard domain extension is a positive legitimacy signal"))
                score -= 5
                break

    # ── 5. URL ENTROPY (randomness detection) ─────────────────
    techniques.append("URL Entropy Analysis")
    ent = _entropy(domain_name)
    if ent > 3.8:
        flags.append(("🟡", f"High domain entropy ({ent:.2f} bits) — looks auto-generated",
                       "Random-looking domain names (e.g. xjq72earn.com) are common in scam campaigns"))
        score += 28
    elif ent > 3.2:
        flags.append(("🟡", f"Moderate domain entropy ({ent:.2f} bits)",
                       "Slightly random domain name — warrants caution"))
        score += 12

    # ── 6. SUSPICIOUS KEYWORDS IN URL ─────────────────────────
    techniques.append("Keyword Fraud Scoring")
    clean_u = u.replace("-", "").replace("_", "").replace(".", "")
    for kw in SCAM_URL_KEYWORDS:
        kw_clean = kw.replace("-", "")
        if kw_clean in clean_u:
            flags.append(("🔴", f"Scam keyword in URL: '{kw}'",
                           "Money/earn/instant keywords in domain names are very strong fraud signals"))
            score += 40
            break

    # ── 7. SUBDOMAIN DEPTH ────────────────────────────────────
    techniques.append("Subdomain Depth Analysis")
    dot_count = full_domain.count(".")
    if dot_count >= 4:
        flags.append(("🔴", f"Excessive subdomain depth ({dot_count} levels)",
                       "e.g. hr.india.earn.jobs.xyz — deep nesting mimics real sites to confuse victims"))
        score += 35
    elif dot_count == 3:
        flags.append(("🟡", f"Unusual subdomain depth ({dot_count} levels)",
                       "Three-level subdomains can obscure the real registrant domain"))
        score += 15

    # ── 8. TYPOSQUATTING / BRAND IMPERSONATION ────────────────
    techniques.append("Typosquatting Detection")
    for brand, official in BRAND_DOMAINS.items():
        if brand in full_domain:
            if full_domain not in (official, "www." + official):
                flags.append(("🔴", f"Typosquatting '{brand.title()}' detected",
                               f"'{full_domain}' is NOT '{official}' — classic brand impersonation tactic"))
                score += 70
                break

    # ── 9. DIGIT DENSITY IN DOMAIN ────────────────────────────
    techniques.append("Domain Digit Density")
    digit_count = len(re.findall(r"\d", domain_name))
    if digit_count >= 4:
        flags.append(("🔴", f"High digit count in domain ({digit_count} digits in '{domain_name}')",
                       "Digit-heavy domains are auto-registered for short-lived fraud campaigns"))
        score += 30
    elif digit_count >= 2:
        flags.append(("🟡", f"Digits in domain name ({digit_count} digits)",
                       "Legitimate company names rarely include numbers in their domain"))
        score += 12

    # ── 10. DOMAIN LENGTH ANOMALY ─────────────────────────────
    techniques.append("Domain Length Analysis")
    dlen = len(domain_name)
    if dlen <= 2:
        flags.append(("🔴", f"Very short domain name ('{domain_name}' — {dlen} chars)",
                       "Extremely short domains are suspicious for a company website"))
        score += 25
    elif dlen > 30:
        flags.append(("🟡", f"Unusually long domain name ({dlen} chars)",
                       "Very long domain names often try to stuff keywords to appear legitimate"))
        score += 15

    # ── TOTAL URL LENGTH ──────────────────────────────────────
    if len(raw) > 120:
        flags.append(("🟡", f"Very long URL ({len(raw)} characters)",
                       "Long URLs can hide redirect chains and obscure the real destination"))
        score += 12

    risk  = _risk(min(score, 100))
    return _build_result(score, flags, techniques, full_domain, raw, risk)


def _build_result(score, flags, techniques, domain, raw, risk):
    labels = {
        "high":   "Fraudulent Domain",
        "medium": "Suspicious Domain",
        "low":    "Legitimate Domain"
    }
    summaries = {
        "high":   "⛔  High risk — domain shows strong fraud indicators",
        "medium": "⚠️   Domain has suspicious characteristics",
        "low":    "✅  Domain appears legitimate"
    }
    return {
        "score":      max(min(score, 100), 0),
        "label":      labels.get(risk, "Unknown"),
        "risk_level": risk,
        "summary":    summaries.get(risk, ""),
        "flags":      flags,
        "techniques": techniques,
        "domain":     domain,
        "raw":        raw
    }


# ══════════════════════════════════════════════════════════════
#  COMPANY NAME ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyse_company(company: str) -> dict:
    if not company or not company.strip():
        return {"score": 0, "flags": [], "risk_level": "unknown",
                "summary": "No company name provided", "is_known": False, "raw": ""}

    c  = company.strip()
    cl = c.lower()
    flags = []
    score = 0
    is_known = False

    # 1. Known legitimate company
    for name in KNOWN_LEGIT_COS:
        if name in cl:
            is_known = True
            flags.append(("✅", f"Recognised company: {c}",
                           "Matches a known verified organisation"))
            score -= 20
            break

    # 2. Scam keywords
    matched = [kw for kw in SCAM_COMPANY_KW if kw in cl]
    if matched:
        flags.append(("🔴", f"Scam keywords: {', '.join(matched[:3])}",
                       "Words like 'earn', 'guaranteed', 'instant' are hallmarks of fake job scams"))
        score += 55

    # 3. Brand impersonation
    for brand in BRAND_DOMAINS:
        if brand in cl and not is_known:
            flags.append(("🔴", f"Possible impersonation of '{brand.title()}'",
                           f"Adds words around '{brand}' to seem like the real company"))
            score += 55
            break

    # 4. Vague name patterns
    vague = [
        (r"^(a\s+)?(company|firm|organization|agency|enterprise)s?$", "Generic placeholder name"),
        (r"^[A-Z]{1,4}$", "Only initials — no real identity"),
        (r"^(global|international|india|worldwide)\s+(solutions?|services?|enterprises?)$",
         "Location + generic service word — classic scam structure"),
    ]
    for pat, reason in vague:
        if re.search(pat, cl):
            flags.append(("🟡", reason, "Scam postings frequently use generic company names"))
            score += 28
            break

    # 5. Excessive legal suffixes stacked
    scount = len(re.findall(
        r"\b(pvt|ltd|llp|inc|corp|llc|international|global|india|worldwide|enterprises|group)\b", cl))
    if scount >= 3:
        flags.append(("🟡", f"{scount} legal/geographic suffixes stacked",
                       "e.g. 'Global International Enterprises Pvt Ltd India'"))
        score += 22

    # 6. Numbers in company name
    if re.search(r"\d", c):
        flags.append(("🟡", "Company name contains numbers",
                       "Legitimate company names rarely include digits"))
        score += 12

    # 7. Length
    if len(c) <= 2:
        flags.append(("🔴", "Company name too short", "Not a credible business identity"))
        score += 35

    risk = _risk(max(score, 0))
    summaries = {
        "high":    "⛔  Company name shows serious red flags",
        "medium":  "⚠️   Company name has suspicious characteristics",
        "low":     "✅  Company name appears credible",
        "unknown": "ℹ️   No company name provided"
    }
    return {"score": max(min(score, 100), 0), "flags": flags, "risk_level": risk,
            "summary": summaries[risk], "is_known": is_known, "raw": company}


# ══════════════════════════════════════════════════════════════
#  COMBINED ANALYSIS  (URL + Company only)
# ══════════════════════════════════════════════════════════════

def analyse_all(website: str, company: str) -> dict:
    """
    Run URL analysis (60%) + company analysis (40%).
    Returns combined fraud risk score and overall risk level.
    """
    url_r = analyse_url(website)
    co_r  = analyse_company(company)

    combined = url_r["score"] * 0.60 + co_r["score"] * 0.40

    red_flags = (
        len([f for f in url_r["flags"] if f[0] == "🔴"]) +
        len([f for f in co_r["flags"]  if f[0] == "🔴"])
    )

    overall = (
        "high"   if combined >= 45 or red_flags >= 3 else
        "medium" if combined >= 18 or red_flags >= 1 else
        "low"
    )

    return {
        "url":             url_r,
        "company":         co_r,
        "combined_score":  round(combined, 1),
        "total_red_flags": red_flags,
        "overall_risk":    overall,
    }
