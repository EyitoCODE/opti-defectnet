import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from splits import build_manifest, validate_manifest
from metrics import classification_metrics


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for label in ("pass", "fail"):
            (self.root / label).mkdir()
            for i in range(20):
                (self.root / label / f"{i}.png").write_bytes(f"{label}-{i}".encode())

    def test_stratified_reproducible_disjoint_complete_split(self):
        manifest = build_manifest(self.root, seed=7)
        self.assertEqual(manifest, build_manifest(self.root, seed=7))
        self.assertNotEqual(manifest["splits"], build_manifest(self.root, seed=8)["splits"])
        validate_manifest(manifest, self.root)
        self.assertEqual([len(v) for v in manifest["splits"].values()], [28, 6, 6])
        for split in manifest["splits"].values():
            self.assertEqual(sum(p.startswith("pass/") for p in split), len(split) // 2)

    def test_changed_dataset_is_rejected(self):
        manifest = build_manifest(self.root)
        (self.root / "pass/0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_manifest(manifest, self.root)

    def test_overlap_is_rejected(self):
        manifest = build_manifest(self.root)
        manifest["splits"]["test"][0] = manifest["splits"]["train"][0]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_manifest(manifest, self.root)

    def test_duplicate_content_is_rejected(self):
        (self.root / "fail/0.png").write_bytes((self.root / "pass/0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_manifest(self.root)


class MetricsTests(unittest.TestCase):
    def test_false_negative_and_precision_recall(self):
        metrics = classification_metrics([0, 0, 1, 1, 1], [0, 1, 0, 1, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertAlmostEqual(metrics["defect_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["defect_recall"], 2 / 3)

    def test_no_predicted_defects_has_undefined_precision(self):
        metrics = classification_metrics([0, 1], [0, 0])
        self.assertIsNone(metrics["defect_precision"])
        self.assertEqual(metrics["defect_recall"], 0)
        json.dumps(metrics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
