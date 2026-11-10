"""Select the best validation checkpoint without inspecting the test split."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import FiberDataset
from model import OptiNet
from splits import build_manifest, fingerprint


def train_model(data_dir="data/synthetic_generated", output_dir="models",
                epochs=5, batch_size=32, seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    manifest = build_manifest(data_dir, seed)
    training = FiberDataset(data_dir, True, manifest["splits"]["train"])
    validation = FiberDataset(data_dir, False, manifest["splits"]["validation"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    model, criterion = OptiNet(), nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_loss, history = float("inf"), []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        model.eval()
        val_loss, correct = 0., 0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
        val_loss /= len(validation)
        row = {"epoch": epoch + 1, "train_loss": train_loss / len(training),
               "validation_loss": val_loss, "validation_accuracy": correct / len(validation)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "split_id": fingerprint(manifest),
                        "epoch": epoch + 1, "seed": seed}, output / "opti_net_v1.pth")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print("Saved best validation checkpoint. Test images were not used for model selection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/synthetic_generated")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    train_model(**vars(parser.parse_args()))
