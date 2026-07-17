# Pipeline overview

How a training run flows through this codebase, and what every experiment so far actually used.
Regenerate/update this file by hand whenever you add or rerun an experiment — it is not auto-generated.

## 1. The pipeline (code path)

```
configs/<name>.yaml
        │
        ▼
scripts/train.py  (entrypoint: python -m scripts.train --config configs/<name>.yaml)
        │
        ├─ src/data/build_transform.py   → transform (config["transforms"])
        ├─ src/data/build_dataset.py     → train/val Dataset (config["dataset"])
        │     ├─ src/data/gastrohun_dataset.py         (dataset.name: gastrohun)
        │     └─ src/data/gastrovision_webdataset.py   (dataset.name: gastrovision_webdataset)
        ├─ src/models/build_model.py     → model (config["model"])
        ├─ src/utils/device.py           → cuda if available else cpu
        │
        ▼
src/training/trainer.py  (Trainer.fit)
        │  each epoch: train_one_epoch() + validate()
        │  validate() computes loss, accuracy, macro precision/recall/F1
        │  logs every epoch to TensorBoard, saves best_model.pt on val_acc improvement
        ▼
outputs/experiments/<experiment_name>/
        ├─ config.yaml                 ← exact config used (auto-copied)
        ├─ checkpoints/best_model.pt   ← best val_acc checkpoint
        ├─ history.csv                 ← per-epoch loss/acc/precision/recall/F1
        ├─ loss_curve.png
        ├─ accuracy_curve.png
        ├─ val_prf_curve.png           ← validation precision/recall/F1 curves
        ├─ tensorboard/                ← same metrics, for `tensorboard --logdir outputs/experiments`
        ├─ classification_report.csv   ← from final evaluate_model() pass on val set
        ├─ confusion_matrix.csv/.png
        ├─ predictions.csv
        └─ metrics.json                ← accuracy/precision/recall/F1 (macro + weighted), final numbers
```

SLURM submissions (`run_experiment.sh <config>`) additionally write `outputs/logs/<config-basename>/` with the SLURM stdout/stderr and a duplicate `train_<jobid>.log`.

`outputs/` and `data/` are both gitignored — they never travel with `git push`/`pull`. See `scripts/setup_data_links.sh` and `scripts/setup_env.sh` for one-time per-machine setup.

## 2. Datasets

| Dataset | Loader | Classes | Source config key |
|---|---|---|---|
| **GastroHUN** | `src/data/gastrohun_dataset.py` | 23 | `dataset.name: gastrohun`, images at `data/Labeled_Images_GastroHun`, splits/labels from `data/official_splits_GastroHun/image_classification.csv` (label column: `Complete agreement`) |
| **GastroVision** | `src/data/gastrovision_webdataset.py` | 22 | `dataset.name: gastrovision_webdataset`, shards at `data/Gastrovision_webdataset/{train,val,test}/*.tar` |

## 3. Current configs → what's actually been run

| Config file | experiment_name | Dataset | Model | Epochs | Status | Best val result |
|---|---|---|---|---|---|---|
| `supervised_gastrohun_resnet18.yaml` | exp01_supervised_gastrohun_resnet18_ep20 | GastroHUN | resnet18 | 20 | **not yet run** under this name (see orphaned runs below) | — |
| `supervised_gastrohun_vit_tiny.yaml` | exp02_supervised_gastrohun_vit_tiny_ep20 | GastroHUN | vit_tiny | 20 | **not yet run** under this name (see orphaned runs below) | — |
| `supervised_gastrovision_resnet18.yaml` | exp04_gastrovision_resnet18_ep01 | GastroVision | resnet18 | 1 | done (smoke test) | val_acc 0.525, val_loss 1.44 — no final eval saved |
| `supervised_gastrovision_resnet18_ep20.yaml` | exp05_supervised_gastrovision_resnet18_ep20 | GastroVision | resnet18 | 20 | done | val_acc 0.544, val_loss 2.33 (train_acc 0.987 — **overfitting**) — no final eval saved |
| `supervised_gastrovision_vit_tiny.yaml` | exp03_supervised_gastrovision_vit_tiny_ep01 | GastroVision | vit_tiny | 1 | done (smoke test) | val_acc 0.396, val_loss 1.89 — **full eval saved** (classification_report/confusion_matrix/metrics.json) |
| `supervised_gastrovision_vit_tiny_ep20.yaml` | exp06_supervised_gastrovision_vit_tiny_ep20 | GastroVision | vit_tiny | 20 | done | val_acc 0.544, val_loss 1.35 — no final eval saved |

All models trained `pretrained: false` (from scratch), Adam optimizer, `supervised_basic` transform (resize 224 + ToTensor, no augmentation), seed 42.

## 4. Orphaned results (no matching config file anymore)

These exist in `outputs/experiments/` but their source config was deleted or superseded — kept here so you know what they are instead of wondering:

| Folder | What it was | Epochs | Notes |
|---|---|---|---|
| `exp01_resnet18` | Ran via the old standalone `scripts/train_resnet18_full.py` (deleted) with hardcoded `/home/vasia/Downloads/...` paths | 3/3 | Pre-pipeline, GastroHUN, resnet18 |
| `exp01_resnet18_config` | Ran from `configs/resnet18.yaml` (deleted, superseded by `supervised_gastrohun_resnet18.yaml`) | 20/20 | GastroHUN, resnet18. val_acc 0.678, but **train_loss collapsed to 0.067 — badly overfit** |
| `exp02_vit_tiny` | Ran from `configs/vit_tiny.yaml` (deleted, superseded by `supervised_gastrohun_vit_tiny.yaml`) | 20/20 | GastroHUN, vit_tiny. val_acc 0.641 |
| `exp03_gastrovision_vit_tiny` | Config had `root: Gastrovision_webdataset` (unresolvable path, from before the `data/` symlink fix) | 0/1 | **Crashed before epoch 1** — no history |
| `exp03_gastrovision_vit_tiny_ep01` | Same broken-path issue, earlier attempt | 0/1 | **Crashed before epoch 1** — no history |

**In short:** `exp01_supervised_gastrohun_resnet18_ep20` and `exp02_supervised_gastrohun_vit_tiny_ep20` (today's GastroHUN configs) haven't been run since the pipeline was standardized — the `exp01_resnet18_config`/`exp02_vit_tiny` results are from earlier, now-deleted config versions and shouldn't be treated as validating the current configs.

## 4a. Data-quality notes (checked 2026-07-17)

- **GastroHUN**: mild class imbalance (~2.7–3.1x largest/smallest class in every split), **zero** train/val/test leakage (checked via patient_id + filename). Clean.
- **GastroVision webdataset**: severe class imbalance (17 to 880 samples/class, ~52x) and **real leakage** — 9 identical-key images shared between train/val, 12 between train/test, 4 between val/test (out of ~8k total). Small in magnitude, but any GastroVision-side val metrics (e.g. `exp05`/`exp06`) are very slightly optimistic. Doesn't affect GastroHUN-target experiments below.

## 4b. Training-recipe fixes (added 2026-07-17)

Earlier supervised runs trained with no augmentation, unweighted plain Adam, a single LR for the whole model, and no early stopping — which is why `exp05`/`exp06` overfit badly (train_acc ~0.99 vs val_acc ~0.54). Added, all opt-in / backward compatible:

- `transforms.name: supervised_augmented` — RandomResizedCrop + flip + mild rotation + mild ColorJitter for train only; val/test always stay plain resize+ToTensor regardless of transform name.
- `training.backbone_learning_rate` + `training.head_learning_rate` (set both together) — discriminative LR via `get_param_groups()` in `build_model.py`, so full fine-tuning doesn't wreck pretrained features. Omit both to keep the old single `learning_rate` behavior.
- `training.optimizer: adamw` (already existed, just wasn't used by any config).
- `training.early_stopping_patience: N` — stop after N epochs without val_acc improvement. Omit to train the full epoch count as before.

## 4b-2. Pretrained-backbone normalization fix (added 2026-07-17)

`build_transform.py` only ever did `Resize + ToTensor` — no `Normalize` step — for `supervised_basic`/`supervised_augmented`. That's fine for from-scratch models (`pretrained: false`, no `backbone_checkpoint`), but any config loading a pretrained backbone was feeding it raw `[0,1]` pixels instead of the distribution it actually expects. Confirmed the expected stats genuinely differ per checkpoint (not one universal constant):

- timm ImageNet ViT (`vit_tiny`/`vit_small`/`vit_base` shorthand, `pretrained: true`) → `mean/std = (0.5,0.5,0.5)`
- Meta DINOv2 checkpoints (e.g. `vit_small_patch14_dinov2.lvd142m`) → ImageNet stats `(0.485,0.456,0.406)/(0.229,0.224,0.225)`
- Any `backbone_checkpoint` (your own DINO-pretrained backbone) → also ImageNet stats, since that's what `DINOMultiCropTransform` normalized with during its SSL pretraining
- resnet18/50 `pretrained: true` → ImageNet stats (torchvision default)

Fixed via `resolve_backbone_normalization()` in `build_model.py`, resolved automatically per-model (queried from timm's own `pretrained_cfg` for ViT variants, not hardcoded) and applied in `build_transform.py`. From-scratch configs are completely unaffected — this only changes behavior when `pretrained: true` or `backbone_checkpoint` is set.

**Consequence: `exp08`–`exp11` (DINO-backbone GastroHUN frozen/fine-tune runs) were already executed on the cluster before this fix, with the backbone seeing an unnormalized input it was never calibrated for. Their saved results should be treated as stale/invalid — re-run these configs to get results that actually reflect the pretrained backbone's quality.** Same caution applies to `exp07_imagenet_gastrohun_vit_tiny_ep20` (`imagenet_gastrohun_vit_tiny.yaml`) if it's been run — check before trusting its number.

## 4b-3. Checkpoint/early-stopping criterion + shared label mapping (added 2026-07-17)

Two more bugs found by review:

- **Best checkpoint and early stopping were driven by val_acc, not macro F1.** For a 23-class imbalanced task, accuracy lets common classes dominate the selection criterion, which is the wrong thing to optimize for a "detect pathological findings" task. `Trainer` now tracks `best_val_f1` and both checkpoint-saving and `early_stopping_patience` compare against `val_f1`. `best_model.pt` now also stores `val_f1` alongside `val_acc`. This is a real behavior change for any future run — checkpoint selection may land on a different epoch than before. Doesn't affect already-saved checkpoints, only future/rerun ones.
- **`GastroHUNDataset` built its label→index mapping from each split independently** (`self.labels = sorted(df[label_column].unique())` ran *after* filtering to the split), so train/val/test theoretically didn't have to agree on what index N means. Checked: currently a no-op bug, since all 23 classes happen to appear in every split of the current CSV — but not guaranteed, and exactly the kind of thing that could silently break on the final unpublished evaluation dataset. Fixed by building the label vocabulary from the full (pre-split) CSV once, then filtering to the split afterward. Verified this produces an identical mapping to before on the current data (zero behavior change today, pure future-proofing).

## 4b-4. Four more bugs found by review (added 2026-07-17)

- **GastroVision DINO pretraining never shuffled data.** `build_gastrovision_webdataset()` had `shardshuffle=False` and no `.shuffle()` call, so every epoch iterated the exact same fixed image order every time — a real, silent SSL training-quality bug (already affected `exp07_dino_gastrovision_vit_tiny`). Fixed: `shardshuffle=100` + `.shuffle(1000)` for `split="train"` only; val/test stay deterministic as before.
- **Resuming DINO training with a different `epochs` silently desynced the LR/teacher-momentum schedule.** The schedules are recomputed fresh each invocation, sized to `steps_per_epoch * epochs` — so a resume segment with a different `epochs` than the run that produced the checkpoint got a differently-shaped schedule with no error. The shipped example (`dino_gastrovision_vit_tiny_resume_part1.yaml` epochs=1 → `_part2.yaml` epochs=2) demonstrated exactly this risky pattern — and it's the *only* way to stop a segment early with the old schema, so this wasn't just user error, it was a schema gap. Fixed two ways: (1) added `training.total_epochs` (defaults to `epochs`) as the fixed final-schedule target across every resume segment, separate from `epochs` (how far *this* invocation runs) — `resume_part1`/`resume_part2` now both set `total_epochs: 2`; (2) the checkpoint now records `total_steps`, and resuming raises a clear error if the current config's `total_steps` doesn't match, instead of silently drifting (checkpoints predating this fix have no recorded `total_steps` and just print a warning).
- **Final evaluation reports only showed numeric class indices, never pathology names.** `evaluate_model()` in `metrics.py` now takes an optional `class_names` list; `train.py`/`evaluate_checkpoint.py` build it from `dataset.idx_to_label` when available (GastroHUN) and pass `None` otherwise (GastroVision has no string labels), so `classification_report.csv`/`confusion_matrix.csv`/`.png` are now labeled with real class names (e.g. "A1", "L6") wherever possible, with zero behavior change for GastroVision.
- **Dead code**: removed `src/evaluation/evaluate.py`, an unused duplicate `evaluate_model` with a different signature than the one actually imported anywhere (confirmed via grep — zero importers).

## 4b-5. Frozen backbones were still left in train mode (added 2026-07-17)

`Trainer.train_one_epoch()` called plain `self.model.train()` every epoch, including on a frozen backbone. `requires_grad=False` only stops gradient updates — it doesn't stop BatchNorm from continuing to update its running statistics from each batch, or Dropout from staying active, both of which would otherwise keep changing a "frozen" backbone's behavior despite it never receiving a gradient update. Checked impact on this project specifically: all current/planned frozen configs (`exp08`, `exp12`, `exp14`) use ViT backbones with LayerNorm (mode-invariant) and zero-probability Dropout by default, so this had **no numeric effect on any existing result** — but it would immediately start mattering the moment a frozen ResNet (already supported in `build_model.py`, just unused with `freeze_backbone` so far) or a ViT with nonzero dropout is tried. Fixed generally via `Trainer._set_training_mode()`, which walks the parameter tree and puts any submodule whose entire parameter subtree is frozen into eval mode, regardless of architecture or a "head"/"fc" naming convention. Verified on a synthetic model with BatchNorm + Dropout in a frozen backbone: running stats now correctly stop updating.

## 4c. Phase A — backbone-source comparison on GastroHUN (added 2026-07-17)

Matched-capacity (~22M params) comparison of pretraining source, holding model scale (ViT-S) constant, evaluated by fine-tuning/probing on GastroHUN:

| Config | experiment_name | Backbone source | Adaptation | Status |
|---|---|---|---|---|
| `supervised_gastrohun_vit_small_imagenet_frozen.yaml` | exp12 | ImageNet-supervised ViT-S/16 (`vit_small`, timm) | frozen + linear probe | not yet run |
| `supervised_gastrohun_vit_small_imagenet_finetune.yaml` | exp13 | ImageNet-supervised ViT-S/16 | full fine-tune (discriminative LR) | not yet run |
| `supervised_gastrohun_vit_small_dinov2_frozen.yaml` | exp14 | Meta DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`, timm) | frozen + linear probe | not yet run |
| `supervised_gastrohun_vit_small_dinov2_finetune.yaml` | exp15 | Meta DINOv2 ViT-S/14 | full fine-tune (discriminative LR) | not yet run |

Note: ImageNet arm is patch16 (21.7M params), DINOv2 arm is patch14 (22.1M params) — patch size differs because it's intrinsic to each method's official release, but total capacity is matched within ~2%.

Secondary, cheaper vit_tiny (patch16, ~5.7M params) row already exists from earlier work — not capacity-matched to the above, useful only as a "does the effect hold at smaller scale too" check: `imagenet_gastrohun_vit_tiny.yaml` (exp07, ImageNet-pretrained), `dino_gastrohun_vit_tiny_frozen.yaml` (exp08) / `dino_gastrohun_vit_tiny_finetune.yaml` (exp09, + `_lr1e4` variant exp10, + teacher-backbone variant exp11) — all using the from-scratch DINO backbone pretrained on GastroVision (`exp07_dino_gastrovision_vit_tiny`), predating the recipe fixes in 4b (no augmentation, plain Adam, no discriminative LR).

Not yet built: the domain-continued-pretraining branch (Meta DINOv2 → continued DINO-style SSL on GastroVision → GastroHUN), which is the most novel part of the comparison — requires new SSL pretraining code/config, to be done as its own step.

## 5. How to run / monitor

```bash
# one-time per machine
scripts/setup_data_links.sh <path-to-Datasets-folder>
scripts/setup_env.sh

# run one experiment
python -m scripts.train --config configs/<name>.yaml
# or on the cluster via SLURM:
sbatch run_experiment.sh configs/<name>.yaml

# monitor (live or after the fact), across all experiments at once
tensorboard --logdir outputs/experiments
```
