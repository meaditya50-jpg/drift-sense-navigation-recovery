# Drift-Sense: Navigation-Error Recovery

## Overview

Drift-Sense is a deep-learning approach for navigation-error recovery in semiconductor wafer inspection.

The system receives:

- Reference image: 100×100 pixels
- Search image: 1000×1000 pixels
- Target pattern: approximately 10×10 pixels within the search image

The system predicts the target centre `(x, y)` in the search image.

For periodic DRAM layouts, multiple visually similar candidates may exist. The proposed system uses confidence-gated refinement and centre-priority selection.

---

## Selected Architecture

**DRAM-style synthetic architecture**

The synthetic layout contains periodic word-lines, bit-lines and contact/via structures designed to reproduce the localization ambiguity of highly regular semiconductor layouts.

---

## Method

The final model is a lightweight **reference-guided adaptive-zoom CNN**.

Pipeline:

1. Reference/search image preprocessing
2. Scale-aware coarse localization
3. Top-K candidate generation
4. 160×160 ROI extraction
5. Fine reference matching
6. Confidence-gated refinement
7. Centre-priority candidate selection
8. Final `(x, y)` prediction

---

## Repository Structure

```text
drift-sense-navigation-recovery/
│
├── README.md
├── requirements.txt
│
├── inference/
│   └── inference.py
│
├── model/
│   └── best.pt
│
├── dataset/
│   ├── generate_gemini_dram.py
│   └── generate_our_dram.py
│
├── training/
│   ├── train_zoom_drift_sense_v3.py
│   └── train_siamese_correlation.py
│
├── evaluation/
│   ├── evaluate_v3_test.py
│   ├── benchmark_inference.py
│   └── results_summary.md
│
├── examples/
│   ├── success_case.png
│   └── failure_case.png
│
├── references/
│   └── REFERENCES.md
│
└── presentation/
    └── DriftSense_Submission.pdf
