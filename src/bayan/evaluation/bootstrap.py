"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Return mean point estimate and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")

    rng = np.random.default_rng(seed)

    point = float(np.mean(values))

    samples = rng.choice(
        values,
        size=(n_boot, values.size),
        replace=True,
    )

    boot_means = samples.mean(axis=1)

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))

    return point, lo, hi


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Return mean paired difference and percentile bootstrap interval."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size != b.size:
        raise ValueError("a and b must have the same length")

    if a.size == 0:
        raise ValueError("inputs must not be empty")

    diffs = a - b
    rng = np.random.default_rng(seed)

    point = float(np.mean(diffs))

    samples = rng.choice(
        diffs,
        size=(n_boot, diffs.size),
        replace=True,
    )

    boot_means = samples.mean(axis=1)

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))

    return point, lo, hi
