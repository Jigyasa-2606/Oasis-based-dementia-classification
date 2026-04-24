# Dementia MRI Baseline Pipeline

Starter pipeline for 4-class dementia stage classification from MRI JPG slices.

**Defaults (honest evaluation):** training splits by **OASIS subject ID** (`--split_mode subject`) so slices from the same patient do not appear in both train and test. Loss uses **inverse-frequency class weights** unless you pass `--no_class_weights`. Evaluation prints **balanced accuracy**, **macro F1**, and a **majority baseline** alongside accuracy.

## Expected data structure

Point `--data_dir` to a folder containing the 4 class folders:

- `Mild Dementia`
- `Moderate Dementia`
- `Non Demented`
- `Very mild Dementia`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

**Recommended (subject split + class weights):**

```bash
python src/train.py \
  --data_dir "./data/Data-3" \
  --split_mode subject \
  --epochs 8 \
  --batch_size 32 \
  --output_dir outputs
```

**Legacy random image split** (can inflate metrics when many slices per patient):

```bash
python src/train.py \
  --data_dir "./data/Data-3" \
  --split_mode slice \
  --epochs 8 \
  --batch_size 32 \
  --output_dir outputs
```

Disable class weights (optional):

```bash
python src/train.py --data_dir "./data/Data-3" --no_class_weights ...
```

`model_bundle.pt` stores `split_mode` and `class_weights` so evaluation matches the training protocol.

Artifacts:
- `outputs/best_model.pt`
- `outputs/model_bundle.pt`
- `outputs/training_artifacts.json`

## Evaluate

If training finished normally, `outputs/model_bundle.pt` exists. If you **only** have `outputs/best_model.pt`, build the bundle once (class order comes from folder names under `data_dir`, same as training):

```bash
python src/pack_model_bundle.py \
  --weights outputs/best_model.pt \
  --data_dir "./data/Data-3" \
  --split_mode subject \
  --output outputs/model_bundle.pt
```

Use `--split_mode slice` and optionally `--no_class_weights_in_training` if that matches how the weights were trained.

Then:

```bash
python src/evaluate.py \
  --data_dir "./data/Data-3" \
  --model_bundle outputs/model_bundle.pt \
  --output_dir outputs
```

By default this scores the **held-out test split** (~15% of images, same `seed=42` as training). Use `--split val` or `--split train` to score those sets instead.

Outputs:
- Summary: **balanced accuracy**, **macro F1**, **majority baseline**, plus per-class precision/recall/F1
- `outputs/confusion_matrix_test.png` (or `_val` / `_train`)
- `outputs/classification_report_test.txt` (matching split)

## Predict on one image

```bash
python src/predict.py \
  --image_path "/path/to/sample.jpg" \
  --model_bundle outputs/model_bundle.pt
```

Binary mapping:
- `Non Demented` -> `Not likely`
- Any other class -> `Dementia likely`

## Limitations

- **Not for clinical use** — research / education demo only.
- **Moderate dementia** has very few subjects; stratified splits fall back to random when a class is too small for sklearn stratification.
- **External validity** (new scanner / site) is unknown until you test on new data.
- Filenames must follow OASIS-style IDs (`OAS1_####_...`) for subject-level splitting; otherwise IDs fall back to the whole stem (weaker grouping).

## Troubleshooting (macOS)

**`ImportError: cannot import name 'Buffer' from 'torch.nn.parameter'`**  
This often appears when `DataLoader` uses **background workers** (`num_workers > 0`): macOS uses `spawn`, which re-imports your script and can surface broken or mixed PyTorch installs.

- Training now defaults to **`--num_workers 0`** (no worker processes). Re-run training without changing anything.
- If it still fails, reinstall a clean pair of wheels:

```bash
pip install --force-reinstall torch torchvision
```
