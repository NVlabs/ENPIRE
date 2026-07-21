"""
Ablation: reflection type → agent success rate over iterations.

Run compared:
  - "text-based reflect"  (1w0jjoej)

Baselines:
  - human code : 14 / 40 = 35.0 %
  - GR00T      : 24 / 40 = 60.0 %
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from base import (
    WandbRunFetcher,
    monotone_frontier_mask,
    sort_by_iter,
    wilson_ci,
    clopper_pearson_ci,
)
import artistic as art

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTITY  = "nv-gear"
PROJECT = "cap-robocasa"

RUNS: dict[str, str] = {
    "text-based reflect": "04acjgpy",
}

ITER_KEY      = "iteration"
SUCCESS_KEY   = "iter/success_rate"
SUCCESSES_KEY = "iter/successes"
N_SEEDS_KEY   = "iter/n_seeds"
RUNTIME_KEY   = "_runtime"

N_TOTAL = 40                          # denominator for baselines
BASELINE_HUMAN  = 14 / N_TOTAL       # 0.350
BASELINE_GROOT  = 24 / N_TOTAL       # 0.600

OUTPUT = Path(__file__).parents[2] / "figures" / "out" / (Path(__file__).stem + ".png")

# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class SuccessRateFetcher(WandbRunFetcher):
    def fetch(self, run_name: str, vis_ci: bool = False, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
        run  = self.get_run(run_name)
        keys = [ITER_KEY, SUCCESS_KEY]
        if vis_ci:
            keys += [SUCCESSES_KEY, N_SEEDS_KEY]
        df = self.get_history(run, keys=keys, optional_keys=[RUNTIME_KEY], use_cache=not refresh)
        cfg = run.config
        meta = {
            "env_name": (cfg.get("env") or {}).get("name", cfg.get("name", "")),
            "n_seeds":  (cfg.get("execution") or {}).get("n_seeds", "?"),
        }
        return sort_by_iter(df, ITER_KEY), meta


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(data: dict[str, pd.DataFrame], metas: dict[str, dict], vis_ci: bool = False) -> None:
    fig, ax = art.figure(w=6.5, h=4.2)

    colors = art.RUN_COLORS

    for (label, df), color in zip(data.items(), colors):
        iters = df[ITER_KEY].to_numpy()
        sr    = df[SUCCESS_KEY].to_numpy()
        hours = df[RUNTIME_KEY].to_numpy() / 3600.0 if RUNTIME_KEY in df.columns else iters
        mask  = monotone_frontier_mask(sr)

        if vis_ci:
            k = df[SUCCESSES_KEY].to_numpy()
            n = df[N_SEEDS_KEY].to_numpy()

            w_lo, w_hi = wilson_ci(sr[mask], n[mask])
            ax.fill_between(
                hours[mask], w_lo, w_hi,
                color=color, alpha=0.13, linewidth=0, zorder=2,
            )

            cp_lo, cp_hi = clopper_pearson_ci(k[mask], n[mask])
            yerr = np.array([sr[mask] - cp_lo, cp_hi - sr[mask]])
            ax.errorbar(
                hours[mask], sr[mask],
                yerr=yerr, fmt="none",
                ecolor=color, elinewidth=0.9,
                capsize=3, capthick=0.9,
                alpha=0.7, zorder=5,
            )

        ax.scatter(
            hours[~mask], sr[~mask],
            color=color, s=20,
            alpha=art.ALPHA_SCATTER, linewidths=0, zorder=3,
        )

        ax.plot(
            hours[mask], sr[mask],
            color=color, linewidth=1.8,
            marker="o", markersize=5,
            markerfacecolor=color, markeredgewidth=0,
            alpha=art.ALPHA_FRONTIER, zorder=6,
            label=label,
        )

        last_h    = hours[mask][-1]
        last_sr   = sr[mask][-1]
        last_iter = int(iters[mask][-1])
        ax.annotate(
            f"iter {last_iter}",
            xy=(last_h, last_sr),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=art.FONTSIZE["annot"],
            color=color,
            va="bottom",
        )

    ax.set_xlabel("Wall time (h)")
    ax.set_ylabel("Success Rate")

    all_sr = np.concatenate([df[SUCCESS_KEY].to_numpy() for df in data.values()])
    if all_sr.max() <= 1.0:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(bottom=0)

    # Top-left: env name + n_seeds
    first_meta = next(iter(metas.values()))
    ax.text(
        0.02, 0.98,
        f"{first_meta['env_name']}    n_seeds = {first_meta['n_seeds']}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=art.FONTSIZE["annot"],
        color=art.PALETTE["slate"],
    )

    ax.axhline(
        BASELINE_HUMAN,
        color=art.PALETTE["slate"],
        linewidth=1.2, linestyle="--", zorder=1,
        label=f"human code ({14}/{N_TOTAL})",
    )
    ax.axhline(
        BASELINE_GROOT,
        color=art.PALETTE["amber"],
        linewidth=1.2, linestyle="--", zorder=1,
        label=f"GR00T ({24}/{N_TOTAL})",
    )

    legend_kw: dict = dict(loc="lower right")
    if vis_ci:
        legend_kw["title"] = "shading = Wilson 95% CI\ncaps = Clopper-Pearson 95% CI"
        legend_kw["title_fontsize"] = 7
    ax.legend(**legend_kw)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT)
    print(f"Saved → {OUTPUT}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-ci",  action="store_true", help="Show Wilson + Clopper-Pearson 95% CIs")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch from wandb")
    args = parser.parse_args()

    fetcher = SuccessRateFetcher(entity=ENTITY, project=PROJECT)
    data:  dict[str, pd.DataFrame] = {}
    metas: dict[str, dict] = {}
    for label, run_name in RUNS.items():
        print(f"Fetching '{label}' ({run_name}) …")
        data[label], metas[label] = fetcher.fetch(run_name, vis_ci=args.vis_ci, refresh=args.refresh)
        print(f"  {len(data[label])} points")
    plot(data, metas, vis_ci=args.vis_ci)


if __name__ == "__main__":
    main()
