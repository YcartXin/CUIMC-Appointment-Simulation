from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote


REPO_DIR = Path(__file__).resolve().parents[1]
DOCS = [
    REPO_DIR / "docs" / "metric_analysis.qmd",
    REPO_DIR / "docs" / "metric_analysis.md",
    REPO_DIR / "docs" / "Class 1 Balking Sensitivity Analysis.qmd",
    REPO_DIR / "docs" / "Class 1 Balking Threshold Analysis.qmd",
]
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


class ReportAssetsTest(unittest.TestCase):
    def test_local_report_images_exist(self) -> None:
        missing = []

        for doc_path in DOCS:
            text = doc_path.read_text(encoding="utf-8")
            for match in IMAGE_PATTERN.finditer(text):
                target = unquote(match.group(1).strip())
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0].split("?", 1)[0]
                asset_path = (doc_path.parent / target).resolve()
                if not asset_path.exists():
                    missing.append(f"{doc_path.relative_to(REPO_DIR)} -> {target}")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

