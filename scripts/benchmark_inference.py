"""Lab 7: CPU latency benchmark over a deterministic sample of production traffic."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("artifacts/topic_classifier")
BENCH_MIX = Path("data/serving/bench_mix.npy")

NUM_THREADS = 4
WARMUP_RUNS = 10
SAMPLE_SIZE = 100


def select_sample(texts):
    """Take an evenly spaced deterministic sample across the full production mix."""
    indices = np.linspace(0, len(texts) - 1, SAMPLE_SIZE, dtype=int)
    return texts[indices]


def prepare_inputs(tokenizer, texts, max_length, padding):
    return [
        tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=padding,
        )
        for text in texts
    ]


def benchmark(model, inputs, warmup_runs=WARMUP_RUNS):
    torch.set_num_threads(NUM_THREADS)
    model.to("cpu")
    model.eval()

    print(f"  warming up with {warmup_runs} requests...", flush=True)

    with torch.inference_mode():
        for item in inputs[:warmup_runs]:
            model(**item)

    print(f"  measuring {len(inputs)} requests...", flush=True)

    latencies = []

    with torch.inference_mode():
        for i, item in enumerate(inputs, start=1):
            start = time.perf_counter()
            model(**item)
            latencies.append((time.perf_counter() - start) * 1000.0)

            if i % 10 == 0:
                print(f"  progress: {i}/{len(inputs)}", flush=True)

    latencies = np.asarray(latencies, dtype=float)

    return {
        "n": len(latencies),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "threads": torch.get_num_threads(),
    }


def print_result(name, result):
    print(f"\n{name}")
    print(f"  n={result['n']}")
    print(f"  p50={result['p50_ms']:.2f} ms")
    print(f"  p99={result['p99_ms']:.2f} ms")
    print(f"  threads={result['threads']}")


def main():
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    torch.set_num_threads(NUM_THREADS)

    full_mix = np.load(BENCH_MIX, allow_pickle=True)
    texts = select_sample(full_mix)

    print("=== Lab 7 CPU Inference Benchmark ===")
    print("Device: CPU")
    print(f"OMP_NUM_THREADS={NUM_THREADS}")
    print(f"Full production mix={len(full_mix)}")
    print(f"Latency sample={len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    print("\nPreparing FP32 baseline inputs (max_length=512, padded=512)...")
    baseline_inputs = prepare_inputs(
        tokenizer,
        texts,
        max_length=512,
        padding="max_length",
    )

    print("\n[1/2] FP32 baseline")
    baseline = benchmark(model, baseline_inputs)

    print_result(
        "FP32 baseline | max_length=512 | padded=512",
        baseline,
    )

    print("\nPreparing free-win inputs (max_length=128, dynamic padding)...")
    free_win_inputs = prepare_inputs(
        tokenizer,
        texts,
        max_length=128,
        padding=False,
    )

    print("\n[2/2] FP32 free-win")
    free_win = benchmark(model, free_win_inputs)

    print_result(
        "FP32 free-win | max_length=128 | dynamic padding",
        free_win,
    )

    print(
        f"\nFree-win p50 speed-up: "
        f"{baseline['p50_ms'] / free_win['p50_ms']:.2f}x"
    )


if __name__ == "__main__":
    main()
