
#!/usr/bin/env python3
"""
Drift-Sense Navigation-Error Recovery
Standalone inference script for hackathon evaluation.

Usage:
    python inference.py reference.png search.png

Output:
    x y

The script automatically loads best.pt from:
    1. ./best.pt
    2. ./model/best.pt
    3. ../model/best.pt

Input expectations:
    reference: 100x100 grayscale/colour image
    search:    1000x1000 grayscale/colour image

The implementation matches the trained Drift-Sense V3 checkpoint
architecture used for the reported test results.
"""

import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

SEARCH_SIZE = 1000
REF_SIZE = 100

COARSE_INPUT = 500
TARGET_SCALE_COARSE = 5

TOP_K = 12
ROI_SIZE = 160

FINE_THRESHOLD = 0.65
MAX_FINE_SHIFT_PX = 32.0


# ============================================================
# DEPTHWISE-SEPARABLE CONV
# ============================================================

class DSConv(nn.Module):

    def __init__(self, cin, cout, stride=1):
        super().__init__()

        self.dw = nn.Conv2d(
            cin,
            cin,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cin,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(cin)

        self.pw = nn.Conv2d(
            cin,
            cout,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.pw(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        return x


# ============================================================
# REFERENCE ENCODER
# ============================================================

class ReferenceEncoder(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.full_branch = nn.Sequential(
            DSConv(1, 16, 2),
            DSConv(16, 32, 2),
            DSConv(32, 48, 1),
            DSConv(48, channels, 1),
        )

        self.target_branch = nn.Sequential(
            DSConv(1, 32, 1),
            DSConv(32, channels, 1),
        )

        self.full_proj = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.target_proj = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

    def forward(self, reference):

        full = self.full_branch(reference)

        target_view = F.interpolate(
            reference,
            size=(TARGET_SCALE_COARSE, TARGET_SCALE_COARSE),
            mode="area"
        )

        target_features = self.target_branch(
            target_view
        )

        target_features = self.target_proj(
            target_features
        )

        full_embedding = F.adaptive_avg_pool2d(
            self.full_proj(full),
            1
        )

        full_embedding = F.normalize(
            full_embedding,
            dim=1
        )

        target_embedding = F.adaptive_avg_pool2d(
            target_features,
            1
        )

        target_embedding = F.normalize(
            target_embedding,
            dim=1
        )

        return (
            full,
            target_features,
            full_embedding,
            target_embedding
        )


# ============================================================
# SEARCH ENCODER
# ============================================================

class SearchEncoder(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.net = nn.Sequential(
            DSConv(1, 16, 2),
            DSConv(16, 32, 1),
            DSConv(32, 48, 1),
            DSConv(48, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# COARSE LOCATOR
# ============================================================

class CoarseLocator(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        # IMPORTANT:
        # These names must match the trained checkpoint exactly.
        self.search_projection = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.target_projection = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.full_reference_projection = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                channels,
                32,
                3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                16,
                3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                1
            )
        )

    def forward(
        self,
        target_features,
        target_embedding,
        full_embedding,
        search_features
    ):

        search_map = F.normalize(
            self.search_projection(search_features),
            dim=1
        )

        target_map = F.normalize(
            self.target_projection(target_features),
            dim=1
        )

        target_template = F.adaptive_avg_pool2d(
            target_map,
            1
        )

        local_correlation = (
            search_map * target_template
        ).sum(
            dim=1,
            keepdim=True
        )

        global_reference = F.normalize(
            self.full_reference_projection(full_embedding),
            dim=1
        )

        global_correlation = (
            search_map * global_reference
        ).sum(
            dim=1,
            keepdim=True
        )

        learned_map = self.head(
            search_features
        )

        return (
            learned_map
            +
            3.0 * local_correlation
            +
            1.0 * global_correlation
        )


# ============================================================
# FINE MATCHER
# ============================================================

class FineMatcher(nn.Module):

    def __init__(self):
        super().__init__()

        self.search_branch = nn.Sequential(
            DSConv(1, 24, 2),
            DSConv(24, 40, 2),
            DSConv(40, 56, 2),
            DSConv(56, 72, 2),
        )

        self.reference_branch = nn.Sequential(
            DSConv(1, 24, 2),
            DSConv(24, 40, 2),
            DSConv(40, 56, 2),
            DSConv(56, 72, 2),
        )

        self.fusion = nn.Sequential(
            DSConv(144, 96, 1),
            DSConv(96, 96, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )

        self.confidence = nn.Linear(64, 1)
        self.offset = nn.Linear(64, 2)

    def forward(self, roi, reference):

        ref = F.interpolate(
            reference,
            size=(10, 10),
            mode="area"
        )

        ref = F.interpolate(
            ref,
            size=(ROI_SIZE, ROI_SIZE),
            mode="bilinear",
            align_corners=False
        )

        search_features = self.search_branch(roi)
        reference_features = self.reference_branch(ref)

        fused = torch.cat(
            [search_features, reference_features],
            dim=1
        )

        fused = self.fusion(fused)
        fused = self.pool(fused).flatten(1)
        fused = self.fc(fused)

        confidence_logits = self.confidence(
            fused
        ).squeeze(1)

        offset = torch.tanh(
            self.offset(fused)
        )

        return confidence_logits, offset


# ============================================================
# COMPLETE V3 MODEL
# ============================================================

class AdaptiveZoomDriftSenseV3(nn.Module):

    def __init__(self):
        super().__init__()

        self.reference_encoder = ReferenceEncoder()
        self.search_encoder = SearchEncoder()
        self.coarse_locator = CoarseLocator()
        self.fine_matcher = FineMatcher()

    def coarse_logits(self, reference, search):

        search_small = F.interpolate(
            search,
            size=(COARSE_INPUT, COARSE_INPUT),
            mode="area"
        )

        (
            _full_ref,
            target_features,
            full_embedding,
            target_embedding
        ) = self.reference_encoder(reference)

        search_features = self.search_encoder(
            search_small
        )

        return self.coarse_locator(
            target_features,
            target_embedding,
            full_embedding,
            search_features
        )


# ============================================================
# INPUT LOADING
# ============================================================

def load_image(path: str) -> torch.Tensor:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = Image.open(path).convert("L")

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def validate_inputs(reference: torch.Tensor, search: torch.Tensor):

    if reference.shape[-2:] != (REF_SIZE, REF_SIZE):
        raise ValueError(
            "Reference image must be "
            f"{REF_SIZE}x{REF_SIZE}, got "
            f"{tuple(reference.shape[-2:])}"
        )

    if search.shape[-2:] != (SEARCH_SIZE, SEARCH_SIZE):
        raise ValueError(
            "Search image must be "
            f"{SEARCH_SIZE}x{SEARCH_SIZE}, got "
            f"{tuple(search.shape[-2:])}"
        )


# ============================================================
# TOP-K CANDIDATES
# ============================================================

def topk_candidates(logits, k=TOP_K):

    _, _, h, w = logits.shape

    flat = logits.flatten(1)

    k = min(
        k,
        flat.shape[1]
    )

    values, indices = torch.topk(
        flat,
        k=k,
        dim=1
    )

    ys = (
        indices // w
    ).float()

    xs = (
        indices % w
    ).float()

    xs /= max(w - 1, 1)
    ys /= max(h - 1, 1)

    centers = torch.stack(
        [xs, ys],
        dim=-1
    )

    return values, centers


# ============================================================
# SAFE ROI EXTRACTION
# ============================================================

def crop_rois(search, centers):

    _, c, h, w = search.shape

    k = centers.shape[1]

    half = ROI_SIZE // 2

    output = []

    for ki in range(k):

        cx = int(
            round(
                float(
                    centers[
                        0,
                        ki,
                        0
                    ].item()
                )
                *
                SEARCH_SIZE
            )
        )

        cy = int(
            round(
                float(
                    centers[
                        0,
                        ki,
                        1
                    ].item()
                )
                *
                SEARCH_SIZE
            )
        )

        x0 = cx - half
        y0 = cy - half

        x1 = x0 + ROI_SIZE
        y1 = y0 + ROI_SIZE

        pl = max(0, -x0)
        pt = max(0, -y0)
        pr = max(0, x1 - w)
        pb = max(0, y1 - h)

        image = search

        if pl or pt or pr or pb:

            image = F.pad(
                image,
                (pl, pr, pt, pb),
                value=0
            )

            x0 += pl
            x1 += pl
            y0 += pt
            y1 += pt

        roi = image[
            :,
            :,
            y0:y1,
            x0:x1
        ]

        if roi.shape[-2:] != (
            ROI_SIZE,
            ROI_SIZE
        ):

            fixed = torch.zeros(
                (
                    1,
                    c,
                    ROI_SIZE,
                    ROI_SIZE
                ),
                dtype=search.dtype,
                device=search.device
            )

            hh = min(
                ROI_SIZE,
                roi.shape[-2]
            )

            ww = min(
                ROI_SIZE,
                roi.shape[-1]
            )

            fixed[
                :,
                :,
                :hh,
                :ww
            ] = roi[
                :,
                :,
                :hh,
                :ww
            ]

            roi = fixed

        output.append(roi)

    return torch.cat(
        output,
        dim=0
    ).unsqueeze(0)


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(xy):

    x = xy[..., 0] * SEARCH_SIZE
    y = xy[..., 1] * SEARCH_SIZE

    return torch.sqrt(
        (x - SEARCH_SIZE / 2.0) ** 2
        +
        (y - SEARCH_SIZE / 2.0) ** 2
    )


# ============================================================
# CENTER-PRIORITY SELECTION
# ============================================================

def select_candidate(
    coarse_centers,
    fine_centers,
    confidence,
    coarse_scores
):

    valid = (
        confidence[0]
        >=
        FINE_THRESHOLD
    )

    if valid.any():

        ids = torch.where(valid)[0]

        refined = fine_centers[
            0,
            ids
        ]

        distances = center_distance(
            refined
        )

        best = ids[
            torch.argmin(distances)
        ]

        return fine_centers[
            0,
            best
        ]

    # No trustworthy fine prediction:
    # fall back to strongest coarse candidate.
    best = torch.argmax(
        coarse_scores[0]
    )

    return coarse_centers[
        0,
        best
    ]


# ============================================================
# SINGLE IMAGE-PAIR INFERENCE
# ============================================================

@torch.no_grad()
def localize(
    model,
    reference,
    search
):

    logits = model.coarse_logits(
        reference,
        search
    )

    coarse_scores, coarse_centers = (
        topk_candidates(
            logits,
            TOP_K
        )
    )

    rois = crop_rois(
        search,
        coarse_centers
    )

    k = rois.shape[1]

    flat_rois = rois.reshape(
        k,
        1,
        ROI_SIZE,
        ROI_SIZE
    )

    flat_reference = reference.expand(
        k,
        1,
        REF_SIZE,
        REF_SIZE
    )

    confidence_logits, offset = (
        model.fine_matcher(
            flat_rois,
            flat_reference
        )
    )

    confidence = torch.sigmoid(
        confidence_logits
    ).reshape(
        1,
        k
    )

    offset = offset.reshape(
        1,
        k,
        2
    )

    shift = (
        offset
        *
        MAX_FINE_SHIFT_PX
        /
        SEARCH_SIZE
    )

    fine_centers = torch.clamp(
        coarse_centers
        +
        shift,
        0.0,
        1.0
    )

    selected = select_candidate(
        coarse_centers,
        fine_centers,
        confidence,
        coarse_scores
    )

    x = float(
        selected[0].item()
        *
        SEARCH_SIZE
    )

    y = float(
        selected[1].item()
        *
        SEARCH_SIZE
    )

    return x, y


# ============================================================
# WEIGHT FILE DISCOVERY
# ============================================================

def find_weights():

    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir / "best.pt",
        script_dir / "model" / "best.pt",
        script_dir.parent / "model" / "best.pt",
    ]

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find best.pt. Put the trained model weights "
        "at ./best.pt or ./model/best.pt relative to inference.py."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python inference.py "
            "<reference_image> <search_image>",
            file=sys.stderr
        )

        sys.exit(2)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    reference = load_image(
        reference_path
    )

    search = load_image(
        search_path
    )

    validate_inputs(
        reference,
        search
    )

    model = AdaptiveZoomDriftSenseV3().to(
        device
    )

    weight_path = find_weights()

    checkpoint = torch.load(
        weight_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    reference = reference.to(
        device
    )

    search = search.to(
        device
    )

    # Warm-up is intentionally NOT used here.
    # Applied Materials should receive actual inference latency
    # from a cold invocation if they benchmark the script.

    with torch.amp.autocast(
        "cuda",
        enabled=device.type == "cuda"
    ):

        x, y = localize(
            model,
            reference,
            search
        )

    # Required output: a single coordinate.
    print(
        f"{x:.3f} {y:.3f}"
    )


if __name__ == "__main__":
    main()
