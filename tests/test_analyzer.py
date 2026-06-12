"""
tests/test_analyzer.py
Unit tests for analyzer.py — URL and company fraud detection.
Run: pytest tests/test_analyzer.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from analyzer import analyse_url, analyse_company, analyse_all


# ══════════════════════════════════════════════════════════════
# analyse_url
# ══════════════════════════════════════════════════════════════

class TestAnalyseUrl:

    def test_empty_url_returns_medium_risk(self):
        r = analyse_url("")
        assert r["risk_level"] == "medium"
        assert r["label"] == "Unverifiable"

    def test_https_legitimate_domain(self):
        r = analyse_url("https://infosys.com")
        assert r["risk_level"] == "low"
        assert r["score"] < 22

    def test_http_penalised(self):
        r_http  = analyse_url("http://example.com")
        r_https = analyse_url("https://example.com")
        assert r_http["score"] > r_https["score"]

    def test_free_hosting_high_risk(self):
        r = analyse_url("https://myjobs.wix.com")
        assert r["risk_level"] == "high"

    def test_suspicious_tld_penalised(self):
        r = analyse_url("https://quickjobs.tk")
        assert r["score"] >= 50

    def test_raw_ip_always_high(self):
        r = analyse_url("http://192.168.1.1/jobs")
        assert r["risk_level"] == "high"

    def test_scam_keyword_in_url(self):
        r = analyse_url("https://earn4u.com/jobs")
        assert r["score"] > 20

    def test_typosquatting_infosys(self):
        r = analyse_url("https://infosys-hr.tk/apply")
        assert r["risk_level"] == "high"

    def test_trusted_tld_reduces_score(self):
        r_com = analyse_url("https://acmecorp.com")
        r_xyz = analyse_url("https://acmecorp.xyz")
        assert r_com["score"] < r_xyz["score"]

    def test_url_without_scheme_handled(self):
        """URLs without http(s):// should not crash."""
        r = analyse_url("example.com")
        assert "risk_level" in r

    def test_score_capped_at_100(self):
        # Worst-case URL
        r = analyse_url("http://192.168.0.1")
        assert r["score"] <= 100

    def test_score_floor_at_0(self):
        r = analyse_url("https://google.com")
        assert r["score"] >= 0

    def test_returns_required_keys(self):
        r = analyse_url("https://example.com")
        for key in ("score", "label", "risk_level", "summary", "flags", "techniques", "domain"):
            assert key in r, f"Missing key: {key}"


# ══════════════════════════════════════════════════════════════
# analyse_company
# ══════════════════════════════════════════════════════════════

class TestAnalyseCompany:

    def test_empty_company(self):
        r = analyse_company("")
        assert r["risk_level"] == "unknown"

    def test_known_legit_company(self):
        r = analyse_company("Infosys Limited")
        assert r["is_known"] is True
        # Known company should not be flagged as high risk
        assert r["risk_level"] != "high"

    def test_scam_keywords_high_risk(self):
        r = analyse_company("Global Earn Network Marketing")
        assert r["risk_level"] == "high"

    def test_brand_impersonation_not_suppressed_by_known_match(self):
        """
        BUG FIX: A name like "Infosys Earn Daily" previously hit
        is_known=True (score -20) which suppressed brand-impersonation
        and scam-keyword penalties. After the fix, scam keywords
        must still be detected even when a known brand name is present.
        """
        r = analyse_company("Infosys Earn Daily Work From Home")
        # Scam keywords ("earn", "work from home") must surface
        scam_flag_found = any(
            "scam" in f[1].lower() or "earn" in f[1].lower()
            for f in r["flags"]
        )
        assert scam_flag_found, (
            "Scam keyword flags should not be suppressed by a known-brand match. "
            f"Got flags: {r['flags']}"
        )

    def test_generic_vague_name(self):
        r = analyse_company("Global Solutions")
        # Should at least be medium risk
        assert r["risk_level"] in ("medium", "high")

    def test_numbers_in_name_flagged(self):
        r = analyse_company("Job4U123 Pvt Ltd")
        flag_texts = [f[1] for f in r["flags"]]
        assert any("digit" in t.lower() or "number" in t.lower() for t in flag_texts)

    def test_returns_required_keys(self):
        r = analyse_company("Acme Corp")
        for key in ("score", "flags", "risk_level", "summary", "is_known", "raw"):
            assert key in r


# ══════════════════════════════════════════════════════════════
# analyse_all (combined)
# ══════════════════════════════════════════════════════════════

class TestAnalyseAll:

    def test_legit_url_and_company(self):
        r = analyse_all("https://infosys.com", "Infosys Limited")
        assert r["overall_risk"] in ("low", "medium")

    def test_high_risk_url_drives_combined_high(self):
        r = analyse_all("http://earn4u.tk", "Global Earn Solutions")
        assert r["overall_risk"] == "high"

    def test_combined_score_in_range(self):
        r = analyse_all("https://example.com", "Example Corp")
        assert 0 <= r["combined_score"] <= 100

    def test_returns_required_keys(self):
        r = analyse_all("https://example.com", "Corp")
        for key in ("url", "company", "combined_score", "total_red_flags", "overall_risk"):
            assert key in r
