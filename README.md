# Dementia MRI Baseline Pipeline

Starter pipeline for 4-class dementia stage classification from MRI JPG slices.

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

```bash
python src/train.py \
  --data_dir "/Users/jigyasaverma/Downloads/Data-3" \
  --epochs 8 \
  --batch_size 32 \
  --output_dir outputs
```

Artifacts:
- `outputs/best_model.pt`
- `outputs/model_bundle.pt`
- `outputs/training_artifacts.json`

## Evaluate

If training finished normally, `outputs/model_bundle.pt` exists. If you **only** have `outputs/best_model.pt`, build the bundle once (class order comes from folder names under `data_dir`, same as training):

```bash
python src/pack_model_bundle.py \
  --weights outputs/best_model.pt \
  --data_dir "/Users/jigyasaverma/Downloads/Data-3" \
  --output outputs/model_bundle.pt
```

Then:

```bash
python src/evaluate.py \
  --data_dir "/Users/jigyasaverma/Downloads/Data-3" \
  --model_bundle outputs/model_bundle.pt \
  --output_dir outputs
```

Outputs:
- Console classification report (precision, recall, f1, accuracy)
- `outputs/confusion_matrix.png`

## Predict on one image

```bash
python src/predict.py \
  --image_path "/path/to/sample.jpg" \
  --model_bundle outputs/model_bundle.pt
```

Binary mapping:
- `Non Demented` -> `Not likely`
- Any other class -> `Dementia likely`

## Troubleshooting (macOS)

**`ImportError: cannot import name 'Buffer' from 'torch.nn.parameter'`**  
This often appears when `DataLoader` uses **background workers** (`num_workers > 0`): macOS uses `spawn`, which re-imports your script and can surface broken or mixed PyTorch installs.

- Training now defaults to **`--num_workers 0`** (no worker processes). Re-run training without changing anything.
- If it still fails, reinstall a clean pair of wheels:

```bash
pip install --force-reinstall torch torchvision
```
