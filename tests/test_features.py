import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.extract import extract_features
from features.unified_extractor import extract_all, ALL_FEATURE_COLS

FEATURE_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_at", "num_question", "num_equals", "num_percent",
    "num_digits_in_domain", "num_digits_in_path", "last_path_segment_is_integer",
    "has_ip", "has_https", "num_subdomains",
    "hostname_length", "path_length", "double_slash",
    "num_suspicious_words",
]


class TestExtractFeatures(unittest.TestCase):

    def test_has_ip_detected(self):
        f = extract_features("http://192.168.0.1/login")
        self.assertEqual(f["has_ip"], 1)

    def test_has_ip_not_set_for_domain(self):
        f = extract_features("https://www.google.com")
        self.assertEqual(f["has_ip"], 0)

    def test_https_flag(self):
        self.assertEqual(extract_features("https://example.com")["has_https"], 1)
        self.assertEqual(extract_features("http://example.com")["has_https"], 0)

    def test_suspicious_words_counted(self):
        f = extract_features("http://secure-login.com/verify")
        self.assertGreaterEqual(f["num_suspicious_words"], 1)

    def test_at_in_url(self):
        self.assertEqual(extract_features("http://user@evil.com")["has_at_in_url"], 1)
        self.assertEqual(extract_features("http://safe.com/page")["has_at_in_url"], 0)

    def test_subdomain_count(self):
        f = extract_features("http://a.b.c.evil.com/page")
        self.assertGreaterEqual(f["num_subdomains"], 2)

    def test_url_length(self):
        url = "http://evil.com/" + "a" * 80
        self.assertGreater(extract_features(url)["url_length"], 80)

    def test_all_feature_cols_present(self):
        f = extract_features("https://www.example.com/path?q=1")
        for col in FEATURE_COLS:
            self.assertIn(col, f, msg=f"Missing feature: {col}")

    def test_extra_features_present(self):
        f = extract_features("https://www.example.com")
        for key in ("is_typosquat", "min_levenshtein", "hostname_entropy",
                    "brand_in_subdomain", "tld_suspicious"):
            self.assertIn(key, f, msg=f"Missing extra feature: {key}")

    def test_tld_suspicious_flagged(self):
        self.assertEqual(extract_features("http://evil.xyz/page")["tld_suspicious"], 1)

    def test_tld_not_suspicious_for_com(self):
        self.assertEqual(extract_features("https://example.com")["tld_suspicious"], 0)


class TestUnifiedExtractor(unittest.TestCase):

    def test_extract_all_includes_all_feature_report_cols(self):
        f = extract_all("http://paypal-verify.xyz/account")
        self.assertGreaterEqual(len(f), len(ALL_FEATURE_COLS))
        for key in ALL_FEATURE_COLS:
            self.assertIn(key, f, msg=f"Missing key for feature report: {key}")

    def test_tld_suspicious_in_unified(self):
        f = extract_all("http://evil.xyz/page")
        self.assertEqual(f.get("tld_suspicious"), 1)


if __name__ == "__main__":
    unittest.main()
