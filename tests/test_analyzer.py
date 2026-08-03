"""
tests/test_analyzer.py
Unit tests for analyzer.py — URL and company fraud detection.
Run: pytest tests/test_analyzer.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from analyzer import analyse_url, analyse_company, analyse_all


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
        r = analyse_url("example.com")
        assert "risk_level" in r
        assert r["risk_level"] in ("low", "medium", "high")

    def test_netlify_not_flagged(self):
        """netlify.app and vercel.app were removed from FREE_HOSTING — should not auto-flag."""
        r = analyse_url("https://myapp.netlify.app")
        # Should not be high risk purely due to netlify hosting
        # (may still be medium if other signals fire, but netlify itself is not a flag)
        assert "netlify" not in [f[1].lower() for f in r["flags"] if "free hosting" in f[1].lower()]

    def test_vercel_not_flagged(self):
        r = analyse_url("https://mysite.vercel.app")
        assert "vercel" not in [f[1].lower() for f in r["flags"] if "free hosting" in f[1].lower()]


class TestAnalyseCompany:

    def test_empty_company(self):
        r = analyse_company("")
        assert r["risk_level"] == "unknown"

    def test_known_legit_company_low_risk(self):
        r = analyse_company("Infosys")
        assert r["is_known"] is True
        assert r["risk_level"] == "low"

    def test_scam_keywords_override_known_brand(self):
        """'Infosys Earn Daily Work From Home' should not be low risk."""
        r = analyse_company("Infosys Earn Daily Work From Home")
        assert r["risk_level"] in ("medium", "high")

    def test_mlm_keywords_detected(self):
        r = analyse_company("Global MLM Network Marketing Pvt Ltd")
        assert r["risk_level"] in ("medium", "high")

    def test_generic_name_flagged(self):
        r = analyse_company("Global Solutions")
        assert r["score"] > 0

    def test_numbers_in_name_flagged(self):
        r = analyse_company("Jobs4U 2024 Pvt Ltd")
        assert any("number" in f[1].lower() for f in r["flags"])

    def test_brand_impersonation_flagged(self):
        r = analyse_company("Wipro Earn Online Work From Home")
        flagged = any("impersonation" in f[1].lower() or "wipro" in f[1].lower() for f in r["flags"])
        assert flagged


class TestAnalyseAll:

    def test_returns_expected_keys(self):
        r = analyse_all("https://infosys.com", "Infosys")
        assert "url" in r
        assert "company" in r
        assert "combined_score" in r
        assert "overall_risk" in r

    def test_combined_risk_high_for_suspicious(self):
        r = analyse_all("http://192.168.1.1/jobs", "Earn Daily Online Jobs")
        assert r["overall_risk"] == "high"

    def test_combined_risk_low_for_legit(self):
        r = analyse_all("https://infosys.com", "Infosys")
        assert r["overall_risk"] == "low"
