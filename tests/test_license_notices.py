# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Offline structural checks, not a legal or release-approval audit."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 "
    "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
DERIVED = (
    "sagn_validation_only.py",
    "post_process_detailed.py",
    "post_process_validation_only.py",
    "post_process_final_once.py",
)
PACKAGES = (
    "torch", "dgl", "ogb", "torch-geometric", "transformers", "numpy",
    "pandas", "scipy", "scikit-learn", "networkx", "matplotlib", "seaborn",
    "safetensors", "tqdm",
)


class LicenseNoticeTests(unittest.TestCase):
    def test_source_headers_and_shebangs(self):
        for path in (ROOT / "scripts").iterdir():
            if path.suffix not in {".py", ".sh"}:
                continue
            with self.subTest(path=path.name):
                lines = path.read_text().splitlines()
                self.assertTrue(lines[0].startswith("#!"))
                self.assertEqual(lines[1], "# " + COPYRIGHT)
                self.assertEqual(lines[2], "# SPDX-License-Identifier: MIT")

    def test_retained_sagn_notices(self):
        license_text = (ROOT / "LICENSES/SAGN_with_SLE-LICENSE.txt").read_text()
        expected = "\n".join(
            "#" + (" " + line if line else "")
            for line in license_text.rstrip().splitlines()
        )
        for filename in DERIVED:
            with self.subTest(filename=filename):
                self.assertIn(expected, (ROOT / "scripts" / filename).read_text())
        patch = (ROOT / "integration/0001-Add-LD-and-EmbeddingGemma-feature-integration.patch").read_text()
        self.assertIn(license_text.rstrip(), patch)
        self.assertEqual(patch.count("+" + "# " + COPYRIGHT), 5)

    def test_license_inventory_files_exist(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
        for package in PACKAGES:
            with self.subTest(package=package):
                filename = f"{package}-LICENSE.txt"
                self.assertTrue((ROOT / "LICENSES" / filename).is_file())
                self.assertIn("LICENSES/" + filename, notices)
        self.assertTrue((ROOT / "LICENSES/MPL-2.0.txt").is_file())
        self.assertIn("reference snapshots", notices)

    def test_project_license_and_contribution_policy(self):
        text = (ROOT / "LICENSE").read_text()
        self.assertIn("Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.", text)
        policy = (ROOT / "CONTRIBUTING.md").read_text()
        self.assertIn("not currently accepting external", policy)
        self.assertIn("No project CLA is required", policy)

    def test_local_document_links_exist(self):
        paths = [
            ROOT / "CONTRIBUTING.md", ROOT / "THIRD_PARTY_NOTICES.md",
            ROOT / "LICENSES/README.md", ROOT / "docs/LICENSING.md",
        ]
        for path in paths:
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists())


if __name__ == "__main__":
    unittest.main()
