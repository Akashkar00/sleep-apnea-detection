# Project Status

Progress against the 9 phases defined in `plan.md`. **4 of 9 phases complete (~44% by phase count)** — see the caveat below on why that understates remaining effort.

```mermaid
flowchart TD
    subgraph Completed["DONE — 4/9 phases"]
        direction TB
        P1["Phase 1<br/>Reproducible setup"]
        P2["Phase 2<br/>Data quality checks<br/>70/70 records verified"]
        P3["Phase 3<br/>Signal preprocessing<br/>34,074 windows extracted"]
        P4["Phase 4<br/>ECG-to-image generation<br/>waveform / STFT / CWT"]
    end

    subgraph Remaining["REMAINING — 5/9 phases"]
        direction TB
        P5["Phase 5<br/>Vision model baselines"]
        P6["Phase 6<br/>CV augmentation"]
        P7["Phase 7<br/>Training &amp; imbalance handling"]
        P8["Phase 8<br/>Evaluation &amp; error analysis"]
        P9["Phase 9<br/>Prediction &amp; demo"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9

    classDef done fill:#c8e6c9,stroke:#2e7d32,color:#1b1b1b;
    classDef todo fill:#eeeeee,stroke:#9e9e9e,color:#1b1b1b;
    class P1,P2,P3,P4 done;
    class P5,P6,P7,P8,P9 todo;
```

```mermaid
pie showData
    title Phases complete vs remaining
    "Complete" : 4
    "Remaining" : 5
```

## Caveat on the percentage

Phase count isn't effort-weighted. Phases 1–4 (done) were mostly one-shot pipeline construction; Phases 5–8 (remaining) involve iterative model training, hyperparameter comparisons across 3 representations, and evaluation — typically the heavier half of a project like this by time spent, even though it's "only" 5 of 9 boxes. Treat ~44% as a phase-count milestone, not a time-remaining estimate.

## What's actually in place

- **Phase 1–2**: `config.yaml`, `uv` environment, git repo, dataset verified (70/70 records, checksums OK)
- **Phase 3**: `src/preprocessing.py`, `src/splits.py`, `src/build_dataset.py` — baseline removal, bandpass filter, artifact clipping, robust normalization, windowing with reject/flag logic; record-level stratified train/val/test split
- **Phase 4**: `src/representations.py`, `src/build_images.py` — waveform/STFT/CWT image generation for all 34,074 windows, `data/images/*.npz`, `notebooks/02_signal_to_image.ipynb`
- **Not started**: Phases 5–9 (`src/models.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`, `app/app.py` don't exist yet)
