# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared utilities for wandb data fetching and figure scripts."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy import stats

CACHE_DIR = Path(__file__).parent / ".cache"


def _cache_path(run_id: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{run_id}_history.parquet"


class WandbRunFetcher(ABC):
    """Abstract base for fetching metrics from a W&B run."""

    def __init__(self, entity: str | None = None, project: str | None = None):
        self.project = project or os.environ.get("WANDB_PROJECT")
        self._api: wandb.Api | None = None
        # Resolve entity: explicit arg > env var > logged-in user's default entity
        self.entity = entity or os.environ.get("WANDB_ENTITY") or self.api.default_entity

    @property
    def api(self) -> wandb.Api:
        if self._api is None:
            self._api = wandb.Api()
        return self._api

    def get_run(self, run_name_or_id: str):
        """Resolve a run by ID or display name.

        Search order:
          1. entity/project/run_id  (fast, exact)
          2. display-name filter within entity/project
          3. scan all projects under entity  (fallback when project is wrong/unknown)
        """
        # 1. Direct ID lookup in the configured project
        if self.project:
            try:
                return self.api.run(f"{self.entity}/{self.project}/{run_name_or_id}")
            except Exception:
                pass
            # 2. Display-name search within configured project
            try:
                runs = list(self.api.runs(
                    f"{self.entity}/{self.project}",
                    filters={"display_name": run_name_or_id},
                ))
                if runs:
                    runs.sort(key=lambda r: r.created_at, reverse=True)
                    return runs[0]
            except Exception:
                pass

        # 3. Scan all accessible projects for this entity
        print(f"  [info] scanning all projects under '{self.entity}' for run '{run_name_or_id}' …")
        for proj in self.api.projects(self.entity):
            try:
                run = self.api.run(f"{self.entity}/{proj.name}/{run_name_or_id}")
                print(f"  [info] found in project '{proj.name}'")
                return run
            except Exception:
                pass

        raise ValueError(
            f"Run '{run_name_or_id}' not found anywhere under entity '{self.entity}'"
        )

    def get_history(
        self,
        run,
        keys: list[str],
        optional_keys: list[str] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download run history for the given metric keys, with optional disk cache.

        Keys in `optional_keys` are fetched when available but never raise if missing.
        """
        all_requested = keys + (optional_keys or [])
        cache = _cache_path(run.id)
        if use_cache and cache.exists():
            df = pd.read_parquet(cache)
            if all(k in df.columns for k in all_requested):
                present = [k for k in all_requested if k in df.columns]
                return df[present].dropna(subset=keys)

        df = run.history(keys=all_requested)
        if use_cache:
            df.to_parquet(cache, index=False)
        missing = set(keys) - set(df.columns)
        if missing:
            all_df = run.history()
            raise KeyError(
                f"Keys not found in run history: {missing}. "
                f"Available: {sorted(c for c in all_df.columns if not c.startswith('_'))}"
            )
        present = [k for k in all_requested if k in df.columns]
        return df[present].dropna(subset=keys)

    @abstractmethod
    def fetch(self, run_name: str) -> pd.DataFrame:
        """Fetch and return a tidy DataFrame for the given run."""
        ...


# ---------------------------------------------------------------------------
# Shared analysis utilities
# ---------------------------------------------------------------------------

def monotone_frontier_mask(values: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True at each index where the value is a new running maximum.
    Points with True form the monotone-increasing envelope; connecting them with
    a line shows the 'best seen so far' progression.
    Assumes values are already sorted by iteration.
    """
    mask = np.zeros(len(values), dtype=bool)
    best = -np.inf
    for i, v in enumerate(values):
        if v > best:
            mask[i] = True
            best = v
    return mask


def sort_by_iter(df: pd.DataFrame, iter_col: str = "iter") -> pd.DataFrame:
    return df.sort_values(iter_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Confidence intervals for binomial proportions
# ---------------------------------------------------------------------------

def wilson_ci(
    p: np.ndarray, n: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval. Returns (lower, upper) clipped to [0, 1]."""
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1)


def clopper_pearson_ci(
    k: np.ndarray, n: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson exact binomial CI. Returns (lower, upper) clipped to [0, 1]."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    lower = np.where(k == 0, 0.0, stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = np.where(k == n, 1.0, stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)
