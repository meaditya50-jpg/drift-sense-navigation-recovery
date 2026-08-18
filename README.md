# Drift-Sense: Navigation-Error Recovery

AI-based localization of a reference site inside a lower-magnification SEM search image for periodic DRAM-style layouts.

## Problem

Given:

- Reference image: 100×100 pixels
- Search image: 1000×1000 pixels
- Target pattern: approximately 10×10 pixels in the search image

The system predicts the target centre `(x, y)` in the search image.

When multiple periodic matches are possible, the candidate closest to the search-image centre is preferred.

## Approach

Drift-Sense uses a reference-guided adaptive-zoom deep-learning architecture:

1. Scale-aware coarse localization
2. Top-K candidate generation
3. Adaptive 160×160 ROI extraction
4. Fine reference matching
5. Confidence-gated refinement
6. Centre-priority candidate selection
7. Final `(x, y)` output

The model contains 124,683 parameters.

## Repository Structure

```text
drift-sense-navigation-recovery/
├── README.md
├── requirements.txt
├── inference/
│   └── inference.py
├── model/
│   └── best.pt
├── dataset/
│   └── generate_dram_dataset.py
├── training/
│   └── train_zoom_drift_sense_v3.py
├── evaluation/
│   ├── evaluate_v3_test.py
│   └── results_summary.md
├── examples/
└── references/
    └── REFERENCES.md
