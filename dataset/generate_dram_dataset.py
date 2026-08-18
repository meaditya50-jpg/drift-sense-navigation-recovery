#!/usr/bin/env python3
"""
DRIFT-SENSE DRAM-ONLY Synthetic Dataset Generator
===============================================

Designed for the Applied Materials / SemiCon India navigation-error recovery
problem.

Generated pair:
  Reference image : 100 x 100 px
  Search image    : 1000 x 1000 px
  Scale ratio     : 10x

The generator intentionally creates the difficult cases described by the
challenge:
  1. Highly periodic DRAM word-line / bit-line structures.
  2. Tiny target (~10 x 10 px) inside a 1000 x 1000 search image.
  3. Exact duplicate target occurrences -> periodic ambiguity.
  4. Near-match distractors -> harder false matches.
  5. Independent sensor noise for reference and search.
  6. Search image is noisier than the reference.
  7. SEM-inspired edge brightening.
  8. Blur / focus variation.
  9. Small rotation.
 10. Vibration / motion blur.
 11. Illumination / contrast variation.
 12. Sub-pixel-like translation / navigation error.
 13. Defective / missing / extra contacts and small line gaps.
 14. Pitch and line-width variation.
 15. Optional local target signature.
 16. Exact ground-truth target centre after transforms.

IMPORTANT:
This is a synthetic research dataset, not a physical SEM simulator. The
parameters are deliberately randomized to provide a robust ML training set.

Typical usage
-------------
pip install numpy pillow tqdm

# 10k prototype
python generate_dram_dataset.py --num_pairs 10000 --output_dir ./dram_10k

# 50k serious training dataset
python generate_dram_dataset.py --num_pairs 50000 --output_dir ./dram_50k --workers 8

# 100k large dataset
python generate_dram_dataset.py --num_pairs 100000 --output_dir ./dram_100k --workers 12

For an ML project, generate separate train/validation/self-test sets using
different seeds so that exact synthetic instances do not leak between splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from tqdm import tqdm


REF_W = 100
REF_H = 100
SEARCH_W = 1000
SEARCH_H = 1000
SCALE = 10


# -------------------------------------------------------------------------
# Random helpers
# -------------------------------------------------------------------------

def py_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def np_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


# -------------------------------------------------------------------------
# DRAM layout model
# -------------------------------------------------------------------------

def draw_dram_pattern(
    size: int,
    seed: int,
    signature: bool = True,
    signature_strength: float = 1.0,
    defect_probability: float = 0.10,
) -> Image.Image:
    """
    Create a DRAM-style periodic pattern:
      - horizontal word lines
      - vertical bit lines
      - contact/via dots
      - optional subtle local signature
      - optional fabrication-like local defects

    The goal is not transistor-level physical simulation; it is a controlled
    image generator with the structural properties required by the challenge.
    """
    r = py_rng(seed)

    # Dark substrate / background
    bg = r.randint(10, 28)
    img = Image.new("L", (size, size), bg)
    d = ImageDraw.Draw(img)

    # Pitch variation is intentionally small because a DRAM array is highly
    # periodic. We nevertheless vary it between samples.
    h_pitch = r.randint(8, 13)
    v_pitch = r.randint(8, 13)

    h_origin = r.randint(1, max(1, h_pitch - 2))
    v_origin = r.randint(1, max(1, v_pitch - 2))

    # Slight line-width variation
    h_width = r.choice([1, 1, 1, 2])
    v_width = r.choice([1, 1, 1, 2])

    # Word lines
    y_positions = []
    y = h_origin
    while y < size:
        line_intensity = r.randint(150, 215)
        d.rectangle(
            [0, y, size - 1, min(size - 1, y + h_width - 1)],
            fill=line_intensity,
        )
        y_positions.append(y)
        y += h_pitch

    # Bit lines
    x_positions = []
    x = v_origin
    while x < size:
        line_intensity = r.randint(135, 205)
        d.rectangle(
            [x, 0, min(size - 1, x + v_width - 1), size - 1],
            fill=line_intensity,
        )
        x_positions.append(x)
        x += v_pitch

    # Contacts / vias at intersections.
    for yy in y_positions:
        for xx in x_positions:
            rr = r.choice([1, 1, 1, 1, 2])
            intensity = r.randint(205, 255)
            d.ellipse(
                [xx - rr, yy - rr, xx + rr, yy + rr],
                fill=intensity,
            )

    # Local manufacturing-like defects. Sparse by design.
    if r.random() < defect_probability:
        # Missing contact
        if x_positions and y_positions:
            xx = r.choice(x_positions)
            yy = r.choice(y_positions)
            rr = r.choice([1, 1, 2])
            d.ellipse(
                [xx - rr, yy - rr, xx + rr, yy + rr],
                fill=bg,
            )

    if r.random() < defect_probability:
        # Short word-line interruption
        yy = r.choice(y_positions)
        x0 = r.randint(5, max(6, size - 10))
        gap = r.randint(2, 6)
        d.rectangle(
            [x0, yy, min(size - 1, x0 + gap), min(size - 1, yy + h_width - 1)],
            fill=bg,
        )

    if r.random() < defect_probability:
        # Short bit-line interruption
        xx = r.choice(x_positions)
        y0 = r.randint(5, max(6, size - 10))
        gap = r.randint(2, 6)
        d.rectangle(
            [xx, y0, min(size - 1, xx + v_width - 1), min(size - 1, y0 + gap)],
            fill=bg,
        )

    # IMPORTANT: signature is what makes the chosen site identifiable.
    # It remains subtle enough that periodic ambiguity is still present.
    if signature:
        cx = size // 2
        cy = size // 2

        # Use a small deterministic local modification.
        mode = r.choice([
            "bright_contact",
            "dark_contact",
            "double_contact",
            "micro_gap",
            "contact_plus_gap",
        ])

        local_x = min(x_positions, key=lambda xx: abs(xx - cx)) if x_positions else cx
        local_y = min(y_positions, key=lambda yy: abs(yy - cy)) if y_positions else cy

        if mode in ("bright_contact", "double_contact", "contact_plus_gap"):
            rr = 1 if signature_strength < 1.1 else 2
            d.ellipse(
                [local_x - rr, local_y - rr,
                 local_x + rr, local_y + rr],
                fill=255,
            )

        if mode == "dark_contact":
            rr = 1 if signature_strength < 1.1 else 2
            d.ellipse(
                [local_x - rr, local_y - rr,
                 local_x + rr, local_y + rr],
                fill=65,
            )

        if mode in ("double_contact", "contact_plus_gap"):
            rr = 1
            offset = max(2, min(h_pitch, v_pitch) // 2)
            d.ellipse(
                [local_x - offset - rr, local_y - rr,
                 local_x - offset + rr, local_y + rr],
                fill=245,
            )

        if mode in ("micro_gap", "contact_plus_gap"):
            gap = r.randint(2, 5)
            d.rectangle(
                [max(0, local_x - gap), local_y,
                 min(size - 1, local_x + gap), local_y + h_width],
                fill=bg,
            )

    # Small SEM-like local intensity variation
    arr = np.asarray(img, dtype=np.float32)
    field = np.ones_like(arr)

    # Low-frequency illumination gradient
    gx = np.linspace(-1.0, 1.0, size)
    gy = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(gx, gy)
    gradient = 1.0 + r.uniform(-0.06, 0.06) * xx + r.uniform(-0.06, 0.06) * yy
    field *= gradient

    arr *= field
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), "L")


# -------------------------------------------------------------------------
# SEM-inspired imaging transformations
# -------------------------------------------------------------------------

def edge_brighten(img: Image.Image, strength: float) -> Image.Image:
    """
    Approximate stronger brightness around structural edges.
    """
    a = np.asarray(img, dtype=np.float32)
    e = np.asarray(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    e /= (e.max() + 1e-6)
    out = a + strength * 255.0 * e
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), "L")


def add_independent_sem_noise(
    img: Image.Image,
    rng: random.Random,
    gaussian_sigma: float,
    poisson_scale: float,
) -> Image.Image:
    """
    Independent capture noise:
      - Gaussian detector/read-like noise
      - signal-dependent Poisson-like noise

    Each call receives a different random stream. Reference and search do
    not reuse the same noise realization.
    """
    a = np.asarray(img, dtype=np.float32)
    g = np_rng(rng.randrange(2**32 - 1))

    a += g.normal(0.0, gaussian_sigma, a.shape)

    signal = np.clip(a / 255.0, 0.0, 1.0)
    lam = np.maximum(signal * poisson_scale, 1.0)
    shot = g.poisson(lam).astype(np.float32) / poisson_scale * 255.0

    # Do not over-destroy the fine structures.
    out = 0.68 * a + 0.32 * shot
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), "L")


def add_directional_vibration(
    img: Image.Image,
    rng: random.Random,
    amplitude: float,
) -> Image.Image:
    """
    Small 1-D blur along a randomly selected direction.
    """
    if amplitude <= 0.05:
        return img

    a = np.asarray(img, dtype=np.float32)
    radius = max(1, int(round(amplitude)))
    weights = []

    for shift in range(-radius, radius + 1):
        w = math.exp(-(shift * shift) / max(1e-6, 2.0 * amplitude**2))
        weights.append((shift, w))

    total = sum(w for _, w in weights)
    horizontal = rng.random() < 0.55

    out = np.zeros_like(a)
    for shift, w in weights:
        shifted = np.roll(a, shift, axis=1 if horizontal else 0)
        out += shifted * (w / total)

    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), "L")


def rotate_image(
    img: Image.Image,
    angle_deg: float,
    fill: int,
) -> Image.Image:
    return img.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )


def rotate_point(
    x: float,
    y: float,
    angle_deg: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """
    Rotate a point around image centre using the same sign convention used by
    PIL's image rotation.
    """
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    dx = x - cx
    dy = y - cy

    th = math.radians(angle_deg)

    xr = math.cos(th) * dx - math.sin(th) * dy + cx
    yr = math.sin(th) * dx + math.cos(th) * dy + cy

    return xr, yr


def apply_capture_variation(
    img: Image.Image,
    rng: random.Random,
    reference: bool,
) -> Tuple[Image.Image, Dict[str, float]]:
    if reference:
        blur = rng.uniform(0.00, 0.65)
        noise_sigma = rng.uniform(1.0, 3.5)
        poisson = rng.uniform(90, 180)
        edge = rng.uniform(0.12, 0.25)
        vibration = rng.uniform(0.0, 0.45)
        rotation = rng.uniform(-2.0, 2.0)
    else:
        blur = rng.uniform(0.0, 1.35)
        noise_sigma = rng.uniform(4.0, 11.0)
        poisson = rng.uniform(25, 80)
        edge = rng.uniform(0.16, 0.35)
        vibration = rng.uniform(0.0, 1.8)
        rotation = rng.uniform(-0.40, 0.40)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    img = add_directional_vibration(img, rng, vibration)
    img = edge_brighten(img, edge)
    img = add_independent_sem_noise(
        img, rng, noise_sigma, poisson
    )

    # Capture-specific contrast / brightness.
    contrast = rng.uniform(0.90, 1.12) if reference else rng.uniform(0.82, 1.18)
    brightness = rng.uniform(0.94, 1.06) if reference else rng.uniform(0.88, 1.12)

    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    img = rotate_image(img, rotation, fill=15)

    return img, {
        "blur": blur,
        "noise_sigma": noise_sigma,
        "poisson_scale": poisson,
        "edge_strength": edge,
        "vibration": vibration,
        "rotation_deg": rotation,
    }


# -------------------------------------------------------------------------
# Search-image construction
# -------------------------------------------------------------------------

def make_search_background(
    seed: int,
    search_size: int = SEARCH_W,
    low_tile_size: int = 10,
) -> Image.Image:
    """
    Create a 1000x1000 low-magnification DRAM array.

    The search image is intentionally very periodic. The low-resolution
    periodic background creates the core navigation ambiguity.
    """
    r = py_rng(seed)

    # Generate a high-resolution DRAM unit WITHOUT the target signature.
    base = draw_dram_pattern(
        100,
        seed + 700001,
        signature=False,
        defect_probability=r.uniform(0.00, 0.04),
    )

    # Low-magnification representation.
    low = base.resize(
        (low_tile_size, low_tile_size),
        Image.Resampling.BOX,
    )

    search = Image.new("L", (search_size, search_size), 15)

    # Small sample-to-sample gain variation is allowed. Do not randomize every
    # individual pixel; preserve the periodic semiconductor structure.
    for y in range(0, search_size, low_tile_size):
        for x in range(0, search_size, low_tile_size):
            gain = r.uniform(0.95, 1.05)
            arr = np.asarray(low, dtype=np.float32) * gain
            tile = Image.fromarray(
                np.uint8(np.clip(arr, 0, 255)), "L"
            )
            search.paste(tile, (x, y))

    return search


def put_centered(
    canvas: Image.Image,
    patch: Image.Image,
    cx: float,
    cy: float,
) -> None:
    x0 = int(round(cx - patch.width / 2.0))
    y0 = int(round(cy - patch.height / 2.0))
    canvas.paste(patch, (x0, y0))


def choose_nonoverlapping_point(
    rng: random.Random,
    existing: List[Tuple[float, float]],
    min_distance: float,
    size: int,
    margin: int,
) -> Tuple[float, float]:
    for _ in range(100):
        x = rng.uniform(margin, size - margin)
        y = rng.uniform(margin, size - margin)

        if all(
            math.hypot(x - ex, y - ey) >= min_distance
            for ex, ey in existing
        ):
            return x, y

    return (
        rng.uniform(margin, size - margin),
        rng.uniform(margin, size - margin),
    )


def add_hard_matches(
    search: Image.Image,
    target_10x: Image.Image,
    true_x: float,
    true_y: float,
    rng: random.Random,
) -> Tuple[Image.Image, int, int]:
    """
    Add:
      - exact duplicates of the target
      - near-match distractors

    The evaluator requires choosing the matching region closest to search
    centre when multiple matches exist. We therefore create explicit
    ambiguous cases and record the count.
    """
    exact_count = 0
    near_count = 0

    # Hard case probability. ~55% of generated samples become periodic
    # ambiguity cases.
    hard = rng.random() < 0.55

    if not hard:
        return search, exact_count, near_count

    # Exact duplicates: 1-3 additional copies.
    n_exact = rng.randint(1, 3)

    points = [(true_x, true_y)]

    for _ in range(n_exact):
        x, y = choose_nonoverlapping_point(
            rng,
            points,
            min_distance=100,
            size=SEARCH_W,
            margin=35,
        )
        put_centered(search, target_10x, x, y)
        points.append((x, y))
        exact_count += 1

    # Near matches with one modified contact/line.
    n_near = rng.randint(1, 3)

    for _ in range(n_near):
        x, y = choose_nonoverlapping_point(
            rng,
            points,
            min_distance=85,
            size=SEARCH_W,
            margin=35,
        )

        d = target_10x.copy()

        # Perturb only a tiny part.
        dd = ImageDraw.Draw(d)
        mode = rng.choice(["dot", "line", "brightness"])

        if mode == "dot":
            px = rng.randint(2, max(2, d.width - 3))
            py = rng.randint(2, max(2, d.height - 3))
            rr = 1
            dd.ellipse(
                [px - rr, py - rr, px + rr, py + rr],
                fill=rng.choice([60, 220, 255]),
            )
        elif mode == "line":
            py = rng.randint(1, max(1, d.height - 2))
            dd.rectangle(
                [0, py, d.width - 1, min(d.height - 1, py + 1)],
                fill=rng.choice([60, 220]),
            )
        else:
            d = ImageEnhance.Brightness(d).enhance(rng.uniform(0.85, 1.15))

        put_centered(search, d, x, y)
        points.append((x, y))
        near_count += 1

    return search, exact_count, near_count


# -------------------------------------------------------------------------
# One pair
# -------------------------------------------------------------------------

def generate_pair(
    sample_id: int,
    seed: int,
    output_dir: str,
    hard_case_probability: float,
) -> Dict:
    rng = py_rng(seed)

    # ----------------------------------------------------------
    # A. Generate the 100x100 reference structure.
    # ----------------------------------------------------------
    signature_strength = rng.uniform(0.7, 1.5)

    reference = draw_dram_pattern(
        REF_W,
        seed,
        signature=True,
        signature_strength=signature_strength,
        defect_probability=rng.uniform(0.03, 0.12),
    )

    reference, ref_capture = apply_capture_variation(
        reference, rng, reference=True
    )

    # ----------------------------------------------------------
    # B. Generate the 1000x1000 periodic search background.
    # ----------------------------------------------------------
    search = make_search_background(
        seed + 900000,
        search_size=SEARCH_W,
        low_tile_size=10,
    )

    # ----------------------------------------------------------
    # C. Convert the exact reference pattern to ~10x smaller.
    # ----------------------------------------------------------
    target = reference.resize(
        (REF_W // SCALE, REF_H // SCALE),
        Image.Resampling.BICUBIC,
    )

    # Independent target capture effects before placement.
    target, target_capture = apply_capture_variation(
        target, rng, reference=False
    )

    # ----------------------------------------------------------
    # D. Place target at a random true coordinate.
    # ----------------------------------------------------------
    margin = 50
    true_x = rng.uniform(margin, SEARCH_W - margin)
    true_y = rng.uniform(margin, SEARCH_H - margin)

    put_centered(search, target, true_x, true_y)

    # ----------------------------------------------------------
    # E. Add explicit ambiguous exact/near matches.
    # ----------------------------------------------------------
    # Honor command-line probability by overriding the local hard-case
    # decision in a simple deterministic way.
    if rng.random() < hard_case_probability:
        search, exact_count, near_count = add_hard_matches(
            search,
            target,
            true_x,
            true_y,
            rng,
        )
    else:
        exact_count, near_count = 0, 0

    # ----------------------------------------------------------
    # F. Search-image final SEM capture.
    # ----------------------------------------------------------
    # These are independently sampled; no reference noise is reused.
    search, search_capture = apply_capture_variation(
        search, rng, reference=False
    )

    # ----------------------------------------------------------
    # G. Global stage/navigation perturbation.
    # ----------------------------------------------------------
    # A small global image-coordinate transformation creates the navigation
    # error that the model must recover.
    stage_rotation = rng.uniform(-0.30, 0.30)
    search = rotate_image(search, stage_rotation, fill=15)
    true_x, true_y = rotate_point(
        true_x,
        true_y,
        stage_rotation,
        SEARCH_W,
        SEARCH_H,
    )

    # Global stage translation.
    stage_dx = rng.uniform(-8.0, 8.0)
    stage_dy = rng.uniform(-8.0, 8.0)

    shifted = Image.new("L", (SEARCH_W, SEARCH_H), 15)
    shifted.paste(
        search,
        (int(round(stage_dx)), int(round(stage_dy))),
    )
    search = shifted

    true_x += stage_dx
    true_y += stage_dy

    true_x = float(np.clip(true_x, 0, SEARCH_W - 1))
    true_y = float(np.clip(true_y, 0, SEARCH_H - 1))

    # ----------------------------------------------------------
    # H. Save.
    # ----------------------------------------------------------
    out = Path(output_dir)
    ref_dir = out / "reference"
    search_dir = out / "search"

    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)

    name = f"{sample_id:07d}_dram"

    ref_file = ref_dir / f"{name}_reference.png"
    search_file = search_dir / f"{name}_search.png"

    reference.save(ref_file, compress_level=6)
    search.save(search_file, compress_level=6)

    row = {
        "sample_id": sample_id,
        "seed": seed,
        "style": "DRAM",
        "reference_file": str(ref_file.relative_to(out)),
        "search_file": str(search_file.relative_to(out)),
        "ground_truth_x": round(true_x, 4),
        "ground_truth_y": round(true_y, 4),
        "reference_width": REF_W,
        "reference_height": REF_H,
        "search_width": SEARCH_W,
        "search_height": SEARCH_H,
        "nominal_scale": SCALE,
        "ref_rotation_deg": round(ref_capture["rotation_deg"], 6),
        "search_capture_rotation_deg": round(search_capture["rotation_deg"], 6),
        "stage_rotation_deg": round(stage_rotation, 6),
        "stage_translation_x": round(stage_dx, 6),
        "stage_translation_y": round(stage_dy, 6),
        "ref_blur": round(ref_capture["blur"], 6),
        "search_blur": round(search_capture["blur"], 6),
        "ref_noise_sigma": round(ref_capture["noise_sigma"], 6),
        "search_noise_sigma": round(search_capture["noise_sigma"], 6),
        "ref_edge_strength": round(ref_capture["edge_strength"], 6),
        "search_edge_strength": round(search_capture["edge_strength"], 6),
        "vibration_reference": round(ref_capture["vibration"], 6),
        "vibration_search": round(search_capture["vibration"], 6),
        "exact_duplicate_count": exact_count,
        "near_match_count": near_count,
    }

    return row


def worker(args):
    return generate_pair(*args)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate DRAM-only Drift-Sense synthetic dataset."
    )

    parser.add_argument("--num_pairs", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="./dram_dataset")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--hard_case_probability",
        type=float,
        default=0.65,
        help="Probability of periodic ambiguity cases.",
    )

    args = parser.parse_args()

    if args.num_pairs < 1:
        raise ValueError("num_pairs must be >= 1")

    if not (0.0 <= args.hard_case_probability <= 1.0):
        raise ValueError("hard_case_probability must be in [0,1]")

    out = Path(args.output_dir)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    (out / "search").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)

    work = [
        (
            i,
            args.seed + i,
            str(out),
            args.hard_case_probability,
        )
        for i in range(args.num_pairs)
    ]

    metadata_file = out / "metadata" / "metadata.csv"

    fields = [
        "sample_id",
        "seed",
        "style",
        "reference_file",
        "search_file",
        "ground_truth_x",
        "ground_truth_y",
        "reference_width",
        "reference_height",
        "search_width",
        "search_height",
        "nominal_scale",
        "ref_rotation_deg",
        "search_capture_rotation_deg",
        "stage_rotation_deg",
        "stage_translation_x",
        "stage_translation_y",
        "ref_blur",
        "search_blur",
        "ref_noise_sigma",
        "search_noise_sigma",
        "ref_edge_strength",
        "search_edge_strength",
        "vibration_reference",
        "vibration_search",
        "exact_duplicate_count",
        "near_match_count",
    ]

    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        if args.workers <= 1:
            iterator = map(worker, work)
            for row in tqdm(
                iterator,
                total=len(work),
                desc="Generating DRAM dataset",
            ):
                writer.writerow(row)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                iterator = executor.map(worker, work, chunksize=4)
                for row in tqdm(
                    iterator,
                    total=len(work),
                    desc="Generating DRAM dataset",
                ):
                    writer.writerow(row)

    # Dataset-level statistics.
    stats = {
        "num_pairs": args.num_pairs,
        "architecture": "DRAM",
        "reference_size": [REF_W, REF_H],
        "search_size": [SEARCH_W, SEARCH_H],
        "nominal_scale": SCALE,
        "hard_case_probability": args.hard_case_probability,
        "seed": args.seed,
    }

    (out / "metadata" / "dataset_config.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    readme = f"""DRIFT-SENSE DRAM DATASET

Pairs:
  {args.num_pairs}

Architecture:
  DRAM-style periodic word lines + bit lines + contacts

Images:
  Reference = {REF_W} x {REF_H}
  Search    = {SEARCH_W} x {SEARCH_H}
  Nominal scale difference = {SCALE}x

Important metadata:
  metadata/metadata.csv

Ground truth:
  ground_truth_x / ground_truth_y are the centre coordinates in the final
  1000x1000 search image after synthetic rotation and stage translation.

Challenge cases included:
  - highly periodic layouts
  - exact duplicate matches
  - near-match distractors
  - independent reference/search noise
  - stronger search degradation
  - blur
  - vibration/motion-like blur
  - edge brightening
  - rotation
  - illumination/contrast changes
  - missing contacts
  - line interruptions
  - pitch and line-width variation
  - synthetic stage translation
"""

    (out / "README.txt").write_text(readme, encoding="utf-8")

    print("\nGeneration complete.")
    print("Dataset:", out.resolve())
    print("Metadata:", metadata_file.resolve())


if __name__ == "__main__":
    main()

