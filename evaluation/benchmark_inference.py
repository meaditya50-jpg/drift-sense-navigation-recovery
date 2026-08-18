
import os
import csv
import time
import random
import statistics

import torch

from inference import (
    AdaptiveZoomDriftSenseV3,
    load_image,
    validate_inputs,
    localize,
    find_weights,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

TEST_CSV = r"D:\SemiconIndia\DriftSense\unified_dataset\test.csv"
NUM_SAMPLES = 50
SEED = 20260818

# Set to False to measure only neural inference after model/input
# loading. Set True for end-to-end single-pair timing.
WARMUP = 3


def load_test_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():

    print("=" * 70)
    print("DRIFT-SENSE INFERENCE BENCHMARK")
    print("=" * 70)

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(TEST_CSV)

    random.seed(SEED)

    rows = load_test_rows(TEST_CSV)

    if len(rows) < NUM_SAMPLES:
        raise ValueError(
            f"Need {NUM_SAMPLES} test samples, found {len(rows)}"
        )

    selected = random.sample(rows, NUM_SAMPLES)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = AdaptiveZoomDriftSenseV3().to(device)

    weight_path = find_weights()

    checkpoint = torch.load(
        weight_path,
        map_location=device
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(f"Warm-up runs: {WARMUP}")

    if WARMUP > 0:

        row = selected[0]

        ref = load_image(row["ref_path"]).to(device)
        search = load_image(row["search_path"]).to(device)

        validate_inputs(ref, search)

        for _ in range(WARMUP):

            with torch.no_grad():

                with torch.amp.autocast(
                    "cuda",
                    enabled=device.type == "cuda"
                ):

                    localize(
                        model,
                        ref,
                        search
                    )

            if device.type == "cuda":
                torch.cuda.synchronize()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    times_ms = []
    errors = []

    print(
        f"\nBenchmarking {NUM_SAMPLES} fresh test pairs..."
    )

    for i, row in enumerate(selected, 1):

        reference = load_image(
            row["ref_path"]
        ).to(device)

        search = load_image(
            row["search_path"]
        ).to(device)

        validate_inputs(
            reference,
            search
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():

            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda"
            ):

                x, y = localize(
                    model,
                    reference,
                    search
                )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        times_ms.append(
            elapsed_ms
        )

        true_x = float(row["x"])
        true_y = float(row["y"])

        error = (
            (x - true_x) ** 2
            +
            (y - true_y) ** 2
        ) ** 0.5

        errors.append(error)

        print(
            f"{i:02d}/{NUM_SAMPLES} | "
            f"Latency = {elapsed_ms:.2f} ms | "
            f"Error = {error:.2f} px"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    p95_ms = sorted(times_ms)[
        int(0.95 * len(times_ms)) - 1
    ]

    mean_error = statistics.mean(errors)
    median_error = statistics.median(errors)

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"Samples                 : {NUM_SAMPLES}"
    )

    print(
        f"Mean latency            : {mean_ms:.3f} ms/pair"
    )

    print(
        f"Median latency          : {median_ms:.3f} ms/pair"
    )

    print(
        f"P95 latency             : {p95_ms:.3f} ms/pair"
    )

    print(
        f"Approx. throughput      : {1000.0 / mean_ms:.2f} pairs/sec"
    )

    print(
        f"Mean localization error: {mean_error:.3f} px"
    )

    print(
        f"Median localization error: {median_error:.3f} px"
    )

    print(
        f"<=10 px accuracy        : "
        f"{sum(e <= 10 for e in errors) / NUM_SAMPLES * 100:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
