# Opti-DefectNet: Synthetic Fiber Inspection Prototype

A PyTorch/OpenCV research prototype for inspecting synthetic optical-fiber end-face
images. A four-stage CNN classifies clean/defective images; a Gradio interface
supports inference and manual feedback logging.

**Scope:** images are procedurally generated. Performance on real microscope
images and manufacturing acceptance criteria have not been validated. Synthetic
test accuracy is not real-fiber accuracy.

## Reproducible workflow

Requires Python 3.10+. Run from the repository root:

```bash
git clone https://github.com/EyitoCODE/opti-defectnet.git
cd opti-defectnet
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/data_generator.py --samples-per-class 1000 --seed 42
python src/train.py --epochs 5 --batch-size 32 --seed 42
python src/evaluate.py
python app/main.py
```

Generation refuses to overwrite existing data by default. To regenerate synthetic
files explicitly pass `--overwrite`, then retrain. Checkpoints from the earlier
pipeline must be retrained before held-out evaluation.

## Evaluation design

- Seeded, stratified **70% train / 15% validation / 15% test** assignment per class;
  integer rounding applies to small datasets.
- Independent datasets apply random augmentation **only to training**.
- Deterministic validation preprocessing; lowest validation loss selects the checkpoint.
- Test images are excluded from training and checkpoint selection.
- SHA-256 fingerprints reject changed datasets and duplicate image content.
- The checkpoint is bound to its split manifest.
- `models/test_metrics.json` contains accuracy, defect precision/recall/F1,
  false negatives and a confusion matrix (rows=actual, columns=predicted).
  Undefined ratios are JSON null.

Do not tune repeatedly on the test results. A new model-development cycle needs
an independently held-out test set. The seeded CPU baseline is reproducible within
a fixed software/hardware environment; cross-version bitwise equivalence is not promised.

## Checks

```bash
python -m unittest discover -s tests -v
```

GitHub Actions checks split integrity and metric arithmetic, then smoke-tests data
generation, one training epoch and held-out evaluation on a tiny synthetic dataset.
The smoke test establishes pipeline execution, not model quality.

## Layout

- `src/data_generator.py`: seeded samples with signed sensor noise.
- `src/splits.py`: dataset fingerprints and disjoint split manifests.
- `src/dataset.py`: separate training/evaluation transformations.
- `src/train.py`: validation-based checkpoint selection.
- `src/evaluate.py`: held-out metrics and checkpoint/manifest checks.
- `app/main.py`: local Gradio interface and CSV/image feedback collection.

Feedback is logged locally in `flagged_feedback/`. Retraining and independent
validation remain explicit steps.

Next steps: a consented real-image benchmark, microscope/lighting robustness
tests, and assessment of false negatives under realistic defect distributions.
