"""Seeded stratified splits and dataset integrity checks (standard library only)."""
import hashlib
import json
import random
from pathlib import Path

CLASSES = {"pass": 0, "fail": 1}


def inventory(data_dir):
    root = Path(data_dir)
    files, digests = {}, set()
    for label in CLASSES:
        paths = sorted(p for p in (root / label).glob("*")
                       if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(paths) < 3:
            raise ValueError(f"Need at least 3 distinct images in {root / label}")
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in digests:
                raise ValueError(f"Duplicate image content: {path}")
            digests.add(digest)
            files[path.relative_to(root).as_posix()] = digest
    return files


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_manifest(data_dir, seed=42):
    files = inventory(data_dir)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label in CLASSES:
        paths = sorted(p for p in files if p.split("/")[0] == label)
        rng.shuffle(paths)
        ntrain, nval = max(1, int(len(paths) * .7)), max(1, int(len(paths) * .15))
        splits["train"].extend(paths[:ntrain])
        splits["validation"].extend(paths[ntrain:ntrain + nval])
        splits["test"].extend(paths[ntrain + nval:])
    return {"version": 1, "seed": seed, "classes": CLASSES,
            "dataset_sha256": fingerprint(files), "splits": splits}


def validate_manifest(manifest, data_dir):
    files = inventory(data_dir)
    if manifest.get("version") != 1 or manifest.get("classes") != CLASSES:
        raise ValueError("Unsupported split manifest")
    if manifest["dataset_sha256"] != fingerprint(files):
        raise ValueError("Dataset changed since training; retrain")
    splits = manifest["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Missing train/validation/test split")
    paths = [p for group in splits.values() for p in group]
    if any(not group for group in splits.values()) or len(paths) != len(set(paths)):
        raise ValueError("Empty or overlapping splits")
    if set(paths) != set(files):
        raise ValueError("Manifest does not cover exactly the current dataset")
    return manifest
