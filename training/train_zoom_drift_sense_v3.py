import os
import csv
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# DRIFT-SENSE ADAPTIVE-ZOOM V3
# ============================================================
#
# V2 fixes the main problem observed in V1:
#
# V1:
#   1000x1000 search
#       -> 250x250
#       -> ~32x32 feature map
#
# The ~10x10 target therefore became too small.
#
# V2:
#   1000x1000 search
#       -> 500x500 coarse input
#       -> 250x250 spatial feature map
#
# Reference:
#   100x100
#       -> explicit 5x5 target-scale representation
#
# Thus the target remains approximately 2-3 coarse cells wide,
# rather than disappearing into a single feature pixel.
#
# Pipeline:
#   Reference + Search
#          |
#          v
#   scale-aware coarse correlation
#          |
#          v
#   top-K candidates
#          |
#          v
#   160x160 zoom ROIs
#          |
#          v
#   fine reference matching
#          |
#          v
#   center-priority candidate selection
#          |
#          v
#        (x,y)
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = r"D:\SemiconIndia\DriftSense\unified_dataset\train.csv"
VAL_CSV = r"D:\SemiconIndia\DriftSense\unified_dataset\validation.csv"

CHECKPOINT_DIR = r"D:\SemiconIndia\DriftSense\checkpoints_zoom_v3"

SEARCH_SIZE = 1000
REF_SIZE = 100

# Keep half-resolution search for coarse stage.
COARSE_INPUT = 500

# 500 -> 250 after one stride-2 layer.
COARSE_MAP = 250

# At 500x500, a 10px target becomes approximately 5px.
TARGET_SCALE_COARSE = 5

TOP_K = 12
ROI_SIZE = 160

BATCH_SIZE = 2
EPOCHS = 30

LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

SEED = 20260816

FINE_THRESHOLD = 0.65

# Maximum fine correction from a coarse candidate.
MAX_FINE_SHIFT_PX = 32.0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# DATASET
# ============================================================

class DriftSenseDataset(Dataset):

    def __init__(self, csv_file):

        self.samples = []

        with open(csv_file, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:
                self.samples.append(row)

        print(
            f"Loaded {len(self.samples):,} samples from {csv_file}"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_gray(path):

        image = Image.open(path).convert("L")

        arr = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, index):

        row = self.samples[index]

        reference = self.load_gray(
            row["ref_path"]
        )

        search = self.load_gray(
            row["search_path"]
        )

        target = torch.tensor(
            [
                float(row["x"]) / SEARCH_SIZE,
                float(row["y"]) / SEARCH_SIZE
            ],
            dtype=torch.float32
        )

        return reference, search, target


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
# SCALE-AWARE REFERENCE ENCODER
# ============================================================

class ReferenceEncoder(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        # Preserve a conventional reference representation.
        self.full_branch = nn.Sequential(
            DSConv(1, 16, 2),       # 100 -> 50
            DSConv(16, 32, 2),     # 50 -> 25
            DSConv(32, 48, 1),
            DSConv(48, channels, 1),
        )

        # Explicit target-scale branch:
        # 100x100 reference -> 5x5 at the 500x500 coarse scale.
        self.target_branch = nn.Sequential(
            DSConv(1, 32, 1),
            DSConv(32, channels, 1),
        )

        self.full_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.target_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
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
# SCALE-AWARE SEARCH ENCODER
# ============================================================

class SearchEncoder(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        # 500 -> 250, then preserve 250x250 resolution.
        self.net = nn.Sequential(
            DSConv(1, 16, 2),
            DSConv(16, 32, 1),
            DSConv(32, 48, 1),
            DSConv(48, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# SCALE-AWARE CORRELATION LOCATOR
# ============================================================

class CoarseLocator(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.search_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.target_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.full_reference_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
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
            self.search_projection(
                search_features
            ),
            dim=1
        )

        target_map = F.normalize(
            self.target_projection(
                target_features
            ),
            dim=1
        )

        # The 5x5 target-scale feature map is collapsed only in the
        # channel sense; the search spatial dimensions remain 250x250.
        target_template = F.adaptive_avg_pool2d(
            target_map,
            1
        )

        local_correlation = (
            search_map
            *
            target_template
        ).sum(
            dim=1,
            keepdim=True
        )

        global_reference = F.normalize(
            self.full_reference_projection(
                full_embedding
            ),
            dim=1
        )

        global_correlation = (
            search_map
            *
            global_reference
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

        # Stronger fine-stage encoder than V2.
        # Search and reference are encoded separately so the network
        # learns correspondence rather than relying only on concatenation.
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

        confidence_logits = self.confidence(fused).squeeze(1)
        offset = torch.tanh(self.offset(fused))

        return confidence_logits, offset


# ============================================================
# COMPLETE MODEL
# ============================================================

class AdaptiveZoomDriftSenseV3(nn.Module):

    def __init__(self):

        super().__init__()

        self.reference_encoder = ReferenceEncoder()
        self.search_encoder = SearchEncoder()
        self.coarse_locator = CoarseLocator()
        self.fine_matcher = FineMatcher()

    def coarse_logits(
        self,
        reference,
        search
    ):

        # 1000x1000 -> 500x500.
        search_small = F.interpolate(
            search,
            size=(COARSE_INPUT, COARSE_INPUT),
            mode="area"
        )

        (
            full_ref,
            target_features,
            full_embedding,
            target_embedding
        ) = self.reference_encoder(
            reference
        )

        search_features = self.search_encoder(
            search_small
        )

        logits = self.coarse_locator(
            target_features,
            target_embedding,
            full_embedding,
            search_features
        )

        return logits


# ============================================================
# GAUSSIAN TARGET HEATMAP
# ============================================================

def gaussian_heatmap(
    target,
    height,
    width,
    sigma=2.0
):

    yy, xx = torch.meshgrid(
        torch.arange(
            height,
            device=target.device,
            dtype=torch.float32
        ),
        torch.arange(
            width,
            device=target.device,
            dtype=torch.float32
        ),
        indexing="ij"
    )

    tx = target[:, 0] * (width - 1)
    ty = target[:, 1] * (height - 1)

    tx = tx[:, None, None]
    ty = ty[:, None, None]

    d2 = (
        (xx[None] - tx) ** 2
        +
        (yy[None] - ty) ** 2
    )

    h = torch.exp(
        -d2 / (2.0 * sigma * sigma)
    )

    h = h / (
        h.amax(
            dim=(1, 2),
            keepdim=True
        )
        + 1e-8
    )

    return h.unsqueeze(1)


# ============================================================
# SOFT ARGMAX
# ============================================================

def soft_argmax(logits):

    b, _, h, w = logits.shape

    p = F.softmax(
        logits.flatten(1),
        dim=1
    )

    yy, xx = torch.meshgrid(
        torch.arange(
            h,
            device=logits.device,
            dtype=torch.float32
        ),
        torch.arange(
            w,
            device=logits.device,
            dtype=torch.float32
        ),
        indexing="ij"
    )

    xx = xx.flatten()
    yy = yy.flatten()

    x = (
        p * xx.unsqueeze(0)
    ).sum(dim=1) / max(
        w - 1,
        1
    )

    y = (
        p * yy.unsqueeze(0)
    ).sum(dim=1) / max(
        h - 1,
        1
    )

    return torch.stack(
        [x, y],
        dim=1
    )


# ============================================================
# TOP-K CANDIDATES
# ============================================================

def topk_candidates(
    logits,
    k=TOP_K
):

    b, _, h, w = logits.shape

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

    xs /= max(
        w - 1,
        1
    )

    ys /= max(
        h - 1,
        1
    )

    centers = torch.stack(
        [xs, ys],
        dim=-1
    )

    return values, centers


# ============================================================
# SAFE ROI EXTRACTION
# ============================================================

def crop_rois(
    search,
    centers
):

    b, c, h, w = search.shape
    k = centers.shape[1]
    half = ROI_SIZE // 2

    output = []

    for bi in range(b):

        rois_b = []

        for ki in range(k):

            cx = int(
                round(
                    float(
                        centers[
                            bi,
                            ki,
                            0
                        ]
                    )
                    *
                    SEARCH_SIZE
                )
            )

            cy = int(
                round(
                    float(
                        centers[
                            bi,
                            ki,
                            1
                        ]
                    )
                    *
                    SEARCH_SIZE
                )
            )

            x0 = cx - half
            y0 = cy - half

            x1 = x0 + ROI_SIZE
            y1 = y0 + ROI_SIZE

            pl = max(
                0,
                -x0
            )

            pt = max(
                0,
                -y0
            )

            pr = max(
                0,
                x1 - w
            )

            pb = max(
                0,
                y1 - h
            )

            image = search[
                bi:bi + 1
            ]

            if (
                pl
                or pt
                or pr
                or pb
            ):

                image = F.pad(
                    image,
                    (
                        pl,
                        pr,
                        pt,
                        pb
                    ),
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

            rois_b.append(
                roi
            )

        output.append(
            torch.cat(
                rois_b,
                dim=0
            )
        )

    return torch.stack(
        output,
        dim=0
    )


# ============================================================
# TRAINING CANDIDATES
# ============================================================

def make_training_candidates(
    target
):

    candidates = []

    positive_candidate = (
        target
        +
        torch.empty_like(
            target
        ).uniform_(
            -32.0 / SEARCH_SIZE,
            32.0 / SEARCH_SIZE
        )
    ).clamp(
        0,
        1
    )

    candidates.append(
        positive_candidate.unsqueeze(1)
    )

    for _ in range(
        TOP_K - 1
    ):

        candidates.append(
            torch.rand_like(
                target
            ).unsqueeze(1)
        )

    return torch.cat(
        candidates,
        dim=1
    )


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(
    xy
):

    x = (
        xy[..., 0]
        *
        SEARCH_SIZE
    )

    y = (
        xy[..., 1]
        *
        SEARCH_SIZE
    )

    return torch.sqrt(
        (
            x - SEARCH_SIZE / 2.0
        ) ** 2
        +
        (
            y - SEARCH_SIZE / 2.0
        ) ** 2
    )


# ============================================================
# COARSE LOSS
# ============================================================

def coarse_loss(
    logits,
    target
):

    target_heat = gaussian_heatmap(
        target,
        logits.shape[-2],
        logits.shape[-1],
        sigma=2.0
    )

    return F.binary_cross_entropy_with_logits(
        logits,
        target_heat
    )


# ============================================================
# FINE LOSS
# ============================================================

def fine_loss(
    confidence_logits,
    offset,
    target,
    centers
):

    gx = target[:, 0] * SEARCH_SIZE
    gy = target[:, 1] * SEARCH_SIZE

    cx = centers[:, 0] * SEARCH_SIZE
    cy = centers[:, 1] * SEARCH_SIZE

    distance = torch.sqrt(
        (gx - cx) ** 2
        +
        (gy - cy) ** 2
        +
        1e-6
    )

    positive = (
        distance <=
        ROI_SIZE * 0.45
    ).float()

    confidence_loss = (
        F.binary_cross_entropy_with_logits(
            confidence_logits,
            positive
        )
    )

    half = ROI_SIZE / 2.0

    true_offset = torch.stack(
        [
            (gx - cx) / half,
            (gy - cy) / half
        ],
        dim=1
    )

    true_offset = torch.clamp(
        true_offset,
        -1,
        1
    )

    mask = positive.unsqueeze(1)

    if mask.sum() > 0:

        regression_loss = (
            F.smooth_l1_loss(
                offset * mask,
                true_offset * mask,
                reduction="sum"
            )
            /
            mask.sum().clamp_min(1.0)
        )

    else:

        regression_loss = torch.zeros(
            (),
            device=target.device
        )

    return (
        confidence_loss
        +
        2.0 * regression_loss
    )


# ============================================================
# CENTER-PRIORITY SELECTION
# ============================================================

def select_candidate_v3(
    coarse_centers,
    fine_centers,
    confidence,
    coarse_scores
):

    b, k, _ = coarse_centers.shape
    selected = []

    for i in range(b):

        valid = confidence[i] >= FINE_THRESHOLD

        if valid.any():

            ids = torch.where(valid)[0]
            refined = fine_centers[i, ids]
            distances = center_distance(refined)

            best = ids[torch.argmin(distances)]

            selected.append(
                fine_centers[i, best]
            )

        else:

            # Gating rule:
            # unreliable fine stage cannot destroy a good coarse match.
            best = torch.argmax(coarse_scores[i])

            selected.append(
                coarse_centers[i, best]
            )

    return torch.stack(selected, dim=0)


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict(
    model,
    reference,
    search
):

    model.eval()

    logits = model.coarse_logits(
        reference,
        search
    )

    coarse_scores, coarse_centers = topk_candidates(
        logits,
        TOP_K
    )

    rois = crop_rois(
        search,
        coarse_centers
    )

    b, k, c, h, w = rois.shape

    flat_rois = rois.reshape(
        b * k,
        c,
        h,
        w
    )

    flat_reference = reference[:, None].expand(
        b,
        k,
        1,
        REF_SIZE,
        REF_SIZE
    ).reshape(
        b * k,
        1,
        REF_SIZE,
        REF_SIZE
    )

    confidence_logits, offset = model.fine_matcher(
        flat_rois,
        flat_reference
    )

    confidence = torch.sigmoid(
        confidence_logits
    ).reshape(b, k)

    offset = offset.reshape(
        b,
        k,
        2
    )

    # Gate the correction spatially: even a confident candidate is
    # allowed to move only a limited distance from the coarse proposal.
    shift = (
        offset
        * MAX_FINE_SHIFT_PX
        / SEARCH_SIZE
    )

    fine_centers = torch.clamp(
        coarse_centers + shift,
        0.0,
        1.0
    )

    final = select_candidate_v3(
        coarse_centers,
        fine_centers,
        confidence,
        coarse_scores
    )

    return (
        final,
        coarse_centers,
        fine_centers,
        confidence
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    final_errors = []
    coarse_best_errors = []

    fine_used = 0
    total_samples = 0

    for reference, search, target in loader:

        reference = reference.to(device, non_blocking=True)
        search = search.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(
            "cuda",
            enabled=device.type == "cuda"
        ):

            prediction, coarse_centers, fine_centers, confidence = predict(
                model,
                reference,
                search
            )

        final_error = (
            prediction - target
        ) * SEARCH_SIZE

        final_error = torch.sqrt(
            (final_error ** 2).sum(dim=1)
        )

        candidate_error = (
            coarse_centers - target[:, None, :]
        ) * SEARCH_SIZE

        candidate_error = torch.sqrt(
            (candidate_error ** 2).sum(dim=2)
        )

        best_coarse = candidate_error.min(dim=1).values

        max_conf = confidence.max(dim=1).values
        fine_used += int(
            (max_conf >= FINE_THRESHOLD).sum().item()
        )
        total_samples += target.shape[0]

        final_errors.extend(final_error.cpu().numpy())
        coarse_best_errors.extend(best_coarse.cpu().numpy())

    final_errors = np.asarray(final_errors)
    coarse_best_errors = np.asarray(coarse_best_errors)

    return {
        "best_coarse_mean": float(coarse_best_errors.mean()),
        "final_mean": float(final_errors.mean()),
        "final_median": float(np.median(final_errors)),
        "final_p90": float(np.percentile(final_errors, 90)),
        "final_p95": float(np.percentile(final_errors, 95)),
        "acc_1": float(np.mean(final_errors <= 1) * 100),
        "acc_3": float(np.mean(final_errors <= 3) * 100),
        "acc_5": float(np.mean(final_errors <= 5) * 100),
        "acc_10": float(np.mean(final_errors <= 10) * 100),
        "fine_used_percent": 100.0 * fine_used / max(total_samples, 1),
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("DRIFT-SENSE ADAPTIVE-ZOOM V3")
    print("GATED FINE REFINEMENT + CENTER PRIORITY")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_dataset = DriftSenseDataset(
        TRAIN_CSV
    )

    val_dataset = DriftSenseDataset(
        VAL_CSV
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    model = AdaptiveZoomDriftSenseV3().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Total parameters:",
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    best_error = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for (
            batch_idx,
            (
                reference,
                search,
                target
            )
        ) in enumerate(
            train_loader
        ):

            reference = reference.to(
                device,
                non_blocking=True
            )

            search = search.to(
                device,
                non_blocking=True
            )

            target = target.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda"
            ):

                # ------------------------------------------------
                # Coarse stage
                # ------------------------------------------------

                coarse_logits = (
                    model.coarse_logits(
                        reference,
                        search
                    )
                )

                loss_coarse = (
                    coarse_loss(
                        coarse_logits,
                        target
                    )
                )

                # ------------------------------------------------
                # Fine-stage training candidates
                # ------------------------------------------------

                candidate_centers = (
                    make_training_candidates(
                        target
                    )
                )

                rois = crop_rois(
                    search,
                    candidate_centers
                )

                b, k, c, h, w = rois.shape

                flat_rois = rois.reshape(
                    b * k,
                    c,
                    h,
                    w
                )

                flat_reference = (
                    reference[:, None]
                    .expand(
                        b,
                        k,
                        1,
                        REF_SIZE,
                        REF_SIZE
                    )
                    .reshape(
                        b * k,
                        1,
                        REF_SIZE,
                        REF_SIZE
                    )
                )

                confidence_logits, offset = (
                    model.fine_matcher(
                        flat_rois,
                        flat_reference
                    )
                )

                confidence_logits = (
                    confidence_logits.reshape(
                        b,
                        k
                    )
                )

                offset = (
                    offset.reshape(
                        b,
                        k,
                        2
                    )
                )

                loss_fine = torch.zeros(
                    (),
                    device=device
                )

                for j in range(k):

                    loss_fine = (
                        loss_fine
                        +
                        fine_loss(
                            confidence_logits[:, j],
                            offset[:, j],
                            target,
                            candidate_centers[:, j]
                        )
                    )

                loss_fine = (
                    loss_fine / k
                )

                total_loss = (
                    loss_coarse
                    +
                    1.5 * loss_fine
                )

            if not torch.isfinite(
                total_loss
            ):

                raise RuntimeError(
                    f"Non-finite loss at batch "
                    f"{batch_idx}"
                )

            scaler.scale(
                total_loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                total_loss.item()
            )

            if (
                batch_idx % 500
                == 0
            ):

                print(
                    f"Epoch {epoch + 1}/"
                    f"{EPOCHS} | "
                    f"Batch {batch_idx}/"
                    f"{len(train_loader)} | "
                    f"Loss "
                    f"{running_loss / (batch_idx + 1):.5f} | "
                    f"Coarse "
                    f"{loss_coarse.item():.5f} | "
                    f"Fine "
                    f"{loss_fine.item():.5f}"
                )

        scheduler.step()

        metrics = evaluate(
            model,
            val_loader,
            device
        )

        print(
            "-" * 70
        )

        print(
            f"Epoch {epoch + 1}"
        )

        print(
            f"Best coarse candidate mean: "
            f"{metrics['best_coarse_mean']:.3f} px"
        )

        print(
            f"Final mean error: "
            f"{metrics['final_mean']:.3f} px"
        )

        print(
            f"Final median: "
            f"{metrics['final_median']:.3f} px"
        )

        print(
            f"P90: "
            f"{metrics['final_p90']:.3f} px"
        )

        print(
            f"P95: "
            f"{metrics['final_p95']:.3f} px"
        )

        print(
            f"<=1 px: "
            f"{metrics['acc_1']:.2f}%"
        )

        print(
            f"<=3 px: "
            f"{metrics['acc_3']:.2f}%"
        )

        print(
            f"<=5 px: "
            f"{metrics['acc_5']:.2f}%"
        )

        print(
            f"<=10 px: "
            f"{metrics['acc_10']:.2f}%"
        )

        print(
            f"Fine stage used: "
            f"{metrics['fine_used_percent']:.2f}%"
        )

        checkpoint = {
            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "metrics":
                metrics
        }

        torch.save(
            checkpoint,
            os.path.join(
                CHECKPOINT_DIR,
                "latest.pt"
            )
        )

        if (
            metrics["final_mean"]
            <
            best_error
        ):

            best_error = (
                metrics["final_mean"]
            )

            torch.save(
                checkpoint,
                os.path.join(
                    CHECKPOINT_DIR,
                    "best.pt"
                )
            )

            print(
                "★ Best model saved."
            )

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**2
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**2
            )

            print(
                f"GPU memory: "
                f"{allocated:.1f} MB allocated / "
                f"{reserved:.1f} MB reserved"
            )

        print(
            "-" * 70
        )

    print(
        "Training complete."
    )


if __name__ == "__main__":
    main()
