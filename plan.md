# Computer-Vision-Oriented Sleep Apnea Detection Plan

## 1. Project objective

Build an educational deep-learning system that classifies each one-minute ECG
segment as:

- `A` — apnea
- `N` — normal

The primary approach treats ECG as a visual pattern-recognition problem. Each
ECG segment will be transformed into one or more image-like representations
(waveform image, spectrogram, and/or continuous wavelet scalogram), then
classified using computer-vision models.

This project is for learning and demonstration only. It is not a medical
device, diagnostic tool, or replacement for polysomnography or a qualified
healthcare professional.

## 2. Dataset and split policy

**Dataset:** [PhysioNet Apnea-ECG Database v1.0.0](https://physionet.org/content/apnea-ecg/1.0.0/)

- 70 single-lead ECG records sampled at 100 Hz.
- Official learning set: `a01`–`a20`, `b01`–`b05`, `c01`–`c10`.
- Official test set: `x01`–`x35`.
- Per-minute `A`/`N` annotations are available for all records used here.
- Extra respiration and SpO2 channels are excluded; ECG only.

Download at runtime with `wfdb`:

```python
import wfdb
wfdb.dl_database("apnea-ecg", dl_dir="data/raw")
```

The official `x01`–`x35` records remain untouched until final evaluation.
Validation is created only from complete records in the learning set. No
windows from the same record may appear in both training and validation/test.

## 3. Computer-vision representation strategy

For every annotated center minute:

1. Extract the center minute plus optional neighboring context, such as
   `[-1, 0, +1]` minutes.
2. Clean the ECG with baseline-wander removal, bandpass filtering, clipping
   of extreme artifacts, and robust amplitude normalization.
3. Generate fixed-size visual inputs:
   - **Waveform image:** ECG plotted as a grayscale or RGB image with a fixed
     time/amplitude layout.
   - **STFT spectrogram:** time-frequency energy image.
   - **CWT scalogram:** multi-resolution frequency representation.
   - **Optional multi-channel image:** waveform, spectrogram, and scalogram
     stacked as three input channels.
4. Resize or crop consistently, preserving temporal structure. Store tensors
   as `.npz` or `.pt` files with the record ID, minute index, label, and
   preprocessing version.

The first experiment should compare representations independently before
combining them. This identifies whether performance comes from morphology,
frequency content, or context.

## 4. Project structure

```text
sleep-apnea-detection/
├── README.md
├── plan.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                  # WFDB files, gitignored
│   ├── processed/            # normalized signal windows, gitignored
│   ├── images/               # waveform/spectrogram/scalogram tensors
│   └── splits/               # record-level split manifests
├── notebooks/
│   ├── 01_fetch_and_explore.ipynb
│   ├── 02_signal_to_image.ipynb
│   ├── 03_train_cv_models.ipynb
│   └── 04_evaluate_and_visualize.ipynb
├── src/
│   ├── fetch_data.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── representations.py    # waveform, STFT, CWT image generation
│   ├── augmentations.py       # training-only image/signal augmentation
│   ├── datasets.py            # PyTorch datasets and record-aware sampling
│   ├── models.py              # CNN, pretrained CNN, ViT, fusion models
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/checkpoints/
├── reports/figures/
├── reports/metrics.json
├── reports/final_report.md
├── app/app.py                 # Streamlit visualization demo
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_representations.py
    └── test_models.py
```

## 5. Implementation phases

### Phase 1 — Reproducible setup

- Create the repository structure and Python environment.
- Install `wfdb`, PyTorch, NumPy, SciPy, scikit-learn, OpenCV/Pillow,
  Matplotlib, PyWavelets, and Streamlit.
- Add `config.yaml` for sampling rate, window length, image size, filter limits,
  representation type, model, seed, batch size, and learning rate.
- Set deterministic seeds and record the exact preprocessing/model config for
  every experiment.

### Phase 2 — Data acquisition and quality checks

- Fetch the database with `wfdb.dl_database`.
- Verify all expected records and annotation files.
- Check sampling rate, signal length, missing values, clipping, and annotation
  alignment.
- Produce per-record class counts and identify highly imbalanced records.
- Keep duplicate or related records documented and avoid mixing related data
  across splits where applicable.

### Phase 3 — Signal preprocessing

- Remove baseline drift with a high-pass filter or robust detrending.
- Apply a conservative ECG bandpass filter, retaining QRS and rhythm content.
- Use robust per-record normalization based on median and interquartile range
  or z-score normalization fitted only on training data.
- Create one-minute center windows, with context as a configurable option.
- Reject or flag unusable windows instead of silently modifying them.

### Phase 4 — ECG-to-image generation

- Implement deterministic waveform rendering with fixed axes and no labels,
  legends, or plot borders that could leak information.
- Implement STFT and CWT generation with fixed frequency limits and image size.
- Normalize image intensity consistently across splits.
- Save a manifest mapping each image to `record_id`, `minute_index`, label,
  representation, and preprocessing version.
- Visually inspect representative `A` and `N` examples and artifact cases.

### Phase 5 — Vision model baselines

Train models in increasing complexity:

1. Logistic regression or a small MLP on handcrafted signal/image features.
2. Small custom 2D CNN trained from scratch.
3. ResNet18/EfficientNet-style CNN with a new binary classification head.
4. Lightweight vision transformer only if the dataset size supports it.
5. Optional late-fusion model combining waveform, spectrogram, and scalogram
   embeddings.

Recommended input shape:

```text
image: [channels, height, width]
channels: 1 for a single representation, 3 for a fused representation
output: apnea probability for the center minute
```

Use transfer learning cautiously: freeze the backbone first, then unfreeze
selected layers with a smaller learning rate. Compare against the custom CNN
so that the final model is not chosen solely because it is larger.

### Phase 6 — Computer-vision augmentation

Apply augmentation only to training samples and only when it preserves the
meaning of the ECG:

- small time translations/crops that retain the target minute;
- mild amplitude scaling and additive noise;
- small contrast/intensity changes for image inputs;
- limited time/frequency masking for spectrograms;
- optional MixUp only after verifying that mixed labels are meaningful.

Do not use arbitrary flips, rotations, aggressive warping, or augmentations
that reverse time or alter ECG morphology unrealistically.

### Phase 7 — Training and imbalance handling

- Use weighted binary cross-entropy or focal loss, with weights computed from
  the training records only.
- Compare weighted sampling against loss weighting.
- Use early stopping on validation apnea sensitivity and PR-AUC.
- Save checkpoints by validation PR-AUC, sensitivity at a selected threshold,
  and balanced accuracy—not raw accuracy alone.
- Track experiment configuration, seed, training curves, and best threshold.

### Phase 8 — Evaluation and visual error analysis

Evaluate once on the official `x01`–`x35` test set after model selection.
Report:

- sensitivity/recall for apnea;
- specificity;
- precision, F1, balanced accuracy, ROC-AUC, and PR-AUC;
- confusion matrix at the selected validation threshold;
- per-record metrics and confidence distributions;
- false-negative and false-positive minutes;
- calibration curve and reliability metrics where feasible.

For computer-vision interpretability, generate Grad-CAM or equivalent
attention overlays for correctly classified and misclassified images. Check
whether the model focuses on ECG content rather than plot borders, padding,
record identity, or rendering artifacts.

### Phase 9 — Prediction and demo

`src/predict.py` should accept a record ID or local WFDB record, generate the
same preprocessing and image representation used during training, and return:

```text
Record: x05
Predicted apnea minutes: 132 / 487 (27.1%)
Representation: CWT scalogram
Model: EfficientNet-B0 vision baseline
Warning: Educational demonstration only; not a medical diagnosis.
```

The Streamlit app should display:

- ECG waveform and generated image representation;
- per-minute apnea probability timeline;
- thresholded predictions and summary statistics;
- optional Grad-CAM overlay;
- ground-truth comparison only when labels are available;
- a persistent medical disclaimer and no default signal-data storage.

## 6. Success criteria

The project is complete when it can:

- reproduce the same train/validation/test records and image tensors;
- train a documented 2D CNN baseline without record leakage;
- compare at least two ECG image representations;
- evaluate on `x01`–`x35` with sensitivity, specificity, F1, PR-AUC, and
  per-record results;
- show representative visual explanations and error cases;
- run inference through the command-line pipeline and Streamlit demo;
- clearly document limitations, dataset bias, label uncertainty, and the
  non-diagnostic nature of the system.

## 7. Initial milestone

```text
Fetch WFDB records
  -> create record-level splits
  -> preprocess ECG windows
  -> generate waveform + spectrogram images
  -> train small 2D CNN baseline
  -> validate without leakage
  -> inspect Grad-CAM and error cases
```

The first implementation should establish a reliable image-generation and
evaluation pipeline before adding transfer learning, multi-representation
fusion, or a vision transformer.
