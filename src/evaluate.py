"""Evaluate only the held-out split bound to the training checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import FiberDataset
from model import OptiNet
from splits import validate_manifest, fingerprint
from metrics import classification_metrics


def evaluate_model(data_dir="data/synthetic_generated", model_dir="models", batch_size=32):
    directory = Path(model_dir)
    manifest = validate_manifest(
        json.loads((directory / "split_manifest.json").read_text()), data_dir)
    checkpoint = torch.load(directory / "opti_net_v1.pth", map_location="cpu", weights_only=True)
    if checkpoint.get("split_id") != fingerprint(manifest):
        raise ValueError("Checkpoint and split manifest do not match; retrain")
    dataset = FiberDataset(data_dir, False, manifest["splits"]["test"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = OptiNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for images, actual in loader:
            predictions.extend(model(images).argmax(1).tolist())
            labels.extend(actual.tolist())
    metrics = classification_metrics(labels, predictions)
    metrics.update({"evaluation_split": "test", "split_id": fingerprint(manifest),
                    "data_scope": "synthetic imagery; real-fiber performance is unvalidated"})
    (directory / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/synthetic_generated")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=32)
    evaluate_model(**vars(parser.parse_args()))
