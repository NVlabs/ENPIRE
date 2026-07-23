# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_METRIC_ORDER = [
    "loop_total_ms",
    "env_obs_phase_ms",
    "pacing_overshoot_ms",
    "pacing_ms",
    "obs_component_span_ms",
    "left_follower_wait_ms",
    "right_follower_wait_ms",
    "left_leader_wait_ms",
    "right_leader_wait_ms",
    "top_camera_get_image_ms",
    "left_camera_get_image_ms",
    "right_camera_get_image_ms",
]


FOCUS_METRICS = [
    "loop_total_ms",
    "env_obs_phase_ms",
    "pacing_overshoot_ms",
    "obs_component_span_ms",
    "left_follower_wait_ms",
    "right_follower_wait_ms",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric_value(metric: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not metric:
        return default
    value = metric.get(key, default)
    if value is None:
        return default
    return float(value)


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f} ms"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "n/a"
    return f"{value:.2f}"


def _ratio(bad: float | None, good: float | None) -> float | None:
    if bad is None or good is None:
        return None
    if good == 0:
        return math.inf if bad > 0 else 1.0
    return bad / good


def _pct_delta(bad: float | None, good: float | None) -> float | None:
    if bad is None or good is None or good == 0:
        return None
    return (bad - good) / good * 100.0


def _metric_stats(summary: dict[str, Any], metric: str) -> dict[str, Any] | None:
    metrics = summary.get("metrics", {})
    value = metrics.get(metric)
    if value is None:
        return None
    return value


def _meta_label(summary: dict[str, Any], fallback: str) -> str:
    meta = summary.get("meta", {})
    return str(meta.get("label") or meta.get("hostname") or fallback)


def _summary_card(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    meta = summary.get("meta", {})
    return {
        "label": label,
        "hostname": meta.get("hostname"),
        "station_key": meta.get("station_key"),
        "platform": meta.get("platform"),
        "python_version": meta.get("python_version"),
        "target_hz": meta.get("target_hz"),
        "iterations_recorded": meta.get("iterations_recorded"),
        "include_leaders": meta.get("include_leaders"),
        "include_followers": meta.get("include_followers"),
        "include_cameras": meta.get("include_cameras"),
        "summary_path": meta.get("summary_path"),
        "created_at": meta.get("created_at"),
    }


def build_report(bad: dict[str, Any], good: dict[str, Any], bad_path: Path, good_path: Path) -> str:
    bad_label = _meta_label(bad, "bad")
    good_label = _meta_label(good, "good")
    bad_card = _summary_card(bad_label, bad)
    good_card = _summary_card(good_label, good)

    metric_rows: list[dict[str, Any]] = []
    all_metric_names = list(dict.fromkeys(DEFAULT_METRIC_ORDER + list(bad.get("metrics", {}).keys()) + list(good.get("metrics", {}).keys())))
    for metric in all_metric_names:
        bad_metric = _metric_stats(bad, metric)
        good_metric = _metric_stats(good, metric)
        if bad_metric is None and good_metric is None:
            continue
        bad_mean = _metric_value(bad_metric, "mean_ms", math.nan)
        good_mean = _metric_value(good_metric, "mean_ms", math.nan)
        bad_p99 = _metric_value(bad_metric, "p99_ms", math.nan)
        good_p99 = _metric_value(good_metric, "p99_ms", math.nan)
        bad_max = _metric_value(bad_metric, "max_ms", math.nan)
        good_max = _metric_value(good_metric, "max_ms", math.nan)
        bad_gt10 = _metric_value(bad_metric, "gt10ms", math.nan)
        good_gt10 = _metric_value(good_metric, "gt10ms", math.nan)
        metric_rows.append(
            {
                "metric": metric,
                "bad": {
                    "mean": bad_mean,
                    "p99": bad_p99,
                    "max": bad_max,
                    "gt10": bad_gt10,
                    "n": _metric_value(bad_metric, "n", math.nan),
                },
                "good": {
                    "mean": good_mean,
                    "p99": good_p99,
                    "max": good_max,
                    "gt10": good_gt10,
                    "n": _metric_value(good_metric, "n", math.nan),
                },
                "ratio_p99": _ratio(bad_p99, good_p99),
                "ratio_max": _ratio(bad_max, good_max),
                "pct_p99": _pct_delta(bad_p99, good_p99),
                "pct_max": _pct_delta(bad_max, good_max),
                "pct_mean": _pct_delta(bad_mean, good_mean),
            }
        )

    ranked_focus = sorted(
        [row for row in metric_rows if row["metric"] in FOCUS_METRICS],
        key=lambda row: (
            -((row["ratio_p99"] or 0.0) if math.isfinite(row["ratio_p99"] or 0.0) else 1e9),
            -(row["bad"]["p99"] if math.isfinite(row["bad"]["p99"]) else 0.0),
        ),
    )

    top_bad = (bad.get("top_outliers") or {})
    top_good = (good.get("top_outliers") or {})

    payload = {
        "bad_card": bad_card,
        "good_card": good_card,
        "bad_path": str(bad_path),
        "good_path": str(good_path),
        "metrics": metric_rows,
        "focus": ranked_focus,
        "top_bad": top_bad,
        "top_good": top_good,
    }

    payload_json = json.dumps(payload, ensure_ascii=True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Station Timing Comparison</title>
  <style>
    :root {{
      --bg: #0b1020;
      --bg2: #10192f;
      --panel: rgba(18, 25, 46, 0.86);
      --panel-strong: rgba(24, 34, 63, 0.95);
      --line: rgba(148, 163, 184, 0.18);
      --text: #e5edf9;
      --muted: #94a3b8;
      --good: #2dd4bf;
      --good2: #14b8a6;
      --bad: #fb7185;
      --bad2: #f43f5e;
      --warn: #f59e0b;
      --accent: #60a5fa;
      --shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
      --radius: 20px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 20%, rgba(96, 165, 250, 0.16), transparent 30%),
        radial-gradient(circle at 85% 10%, rgba(45, 212, 191, 0.11), transparent 24%),
        radial-gradient(circle at 80% 75%, rgba(244, 63, 94, 0.1), transparent 22%),
        linear-gradient(180deg, var(--bg) 0%, #09101d 100%);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 28px 22px 48px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(9, 14, 27, 0.88));
      box-shadow: var(--shadow);
      padding: 28px 28px 22px;
      margin-bottom: 22px;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -60px -80px auto;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(96, 165, 250, 0.16), transparent 70%);
      filter: blur(12px);
      pointer-events: none;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }}
    h1 {{
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      font-size: clamp(28px, 4vw, 48px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      margin: 0 0 12px;
      max-width: 12ch;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 1100px;
      font-size: 15px;
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .summary {{
      grid-column: span 6;
      padding: 18px;
    }}
    .summary h2, .section h2 {{
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: -0.03em;
    }}
    .summary-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .kv {{
      padding: 14px 14px 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .kv .k {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 7px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .kv .v {{
      font-size: 14px;
      line-height: 1.4;
      word-break: break-word;
    }}
    .section {{
      margin-top: 22px;
      padding: 18px;
    }}
    .focus-grid {{
      grid-column: span 12;
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
    }}
    .focus-card {{
      grid-column: span 3;
      padding: 16px 16px 14px;
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 18px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .toolbar-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }}
    .segmented {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .segmented button {{
      appearance: none;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 12px;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }}
    .segmented button:hover {{
      transform: translateY(-1px);
      border-color: rgba(96, 165, 250, 0.35);
    }}
    .segmented button.active {{
      background: rgba(96, 165, 250, 0.16);
      border-color: rgba(96, 165, 250, 0.52);
      color: #dbeafe;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: 1.4fr 0.9fr;
      gap: 16px;
    }}
    .chart-panel {{
      min-height: 360px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.14);
      background: rgba(255, 255, 255, 0.03);
      position: relative;
    }}
    .chart-panel h3 {{
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: -0.03em;
    }}
    .chart-caption {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-bottom: 8px;
    }}
    .chart-svg {{
      width: 100%;
      height: 320px;
      display: block;
      overflow: visible;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .tick-label {{
      fill: rgba(226, 232, 240, 0.72);
      font-size: 11px;
    }}
    .grid-line {{
      stroke: rgba(148, 163, 184, 0.12);
      stroke-width: 1;
    }}
    .diag-line {{
      stroke: rgba(96, 165, 250, 0.42);
      stroke-width: 1.5;
      stroke-dasharray: 5 5;
    }}
    .metric-point {{
      cursor: pointer;
      transition: opacity 140ms ease, transform 140ms ease;
    }}
    .metric-point.dim {{
      opacity: 0.36;
    }}
    .metric-point.selected {{
      filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.28));
    }}
    .heatmap {{
      display: grid;
      grid-template-columns: 180px repeat(4, minmax(90px, 1fr));
      gap: 8px;
      align-items: stretch;
    }}
    .heatmap-head {{
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0 2px 6px;
    }}
    .heatmap-metric {{
      display: flex;
      align-items: center;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      font-size: 12px;
      color: #dbeafe;
      border: 1px solid rgba(148, 163, 184, 0.08);
    }}
    .heat-cell {{
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 14px;
      min-height: 48px;
      font-size: 12px;
      border: 1px solid rgba(148, 163, 184, 0.1);
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease;
    }}
    .heat-cell:hover {{
      transform: translateY(-1px);
      border-color: rgba(255, 255, 255, 0.25);
    }}
    .heat-cell.selected {{
      outline: 2px solid rgba(96, 165, 250, 0.55);
      outline-offset: 1px;
    }}
    .detail-grid {{
      display: grid;
      gap: 10px;
    }}
    .detail-hero {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .detail-stat {{
      padding: 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(148, 163, 184, 0.08);
    }}
    .detail-stat .k {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .detail-stat .v {{
      font-size: 20px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .mini-bars {{
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }}
    .mini-row {{
      display: grid;
      gap: 8px;
      grid-template-columns: 64px 1fr 64px 64px;
      align-items: center;
      font-size: 12px;
    }}
    .mini-track {{
      position: relative;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.12);
    }}
    .mini-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: inherit;
    }}
    .mini-fill.bad {{ background: linear-gradient(90deg, var(--bad2), #fda4af); }}
    .mini-fill.good {{ background: linear-gradient(90deg, var(--good2), #99f6e4); }}
    .tooltip {{
      position: fixed;
      z-index: 50;
      pointer-events: none;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(7, 12, 24, 0.95);
      border: 1px solid rgba(148, 163, 184, 0.18);
      color: var(--text);
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 16px 32px rgba(2, 6, 23, 0.4);
      max-width: 280px;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 100ms ease, transform 100ms ease;
    }}
    .tooltip.visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    details.table-details {{
      margin-top: 16px;
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.02);
      overflow: hidden;
    }}
    details.table-details > summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      font-size: 15px;
    }}
    details.table-details > summary::-webkit-details-marker {{
      display: none;
    }}
    .metric-name {{
      font-size: 13px;
      color: #cbd5e1;
      margin-bottom: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 52px 1fr 52px;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
    }}
    .bar-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .bar-track {{
      position: relative;
      height: 12px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.12);
      overflow: hidden;
    }}
    .bar-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: inherit;
      transition: width 180ms ease;
    }}
    .bar-fill.bad {{ background: linear-gradient(90deg, var(--bad), #ff8fab); }}
    .bar-fill.good {{ background: linear-gradient(90deg, var(--good2), #5eead4); }}
    .delta {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .delta strong {{
      color: var(--text);
    }}
    .delta.bad strong {{ color: var(--bad); }}
    .delta.good strong {{ color: var(--good); }}
    .table-wrap {{
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.16);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: rgba(3, 7, 18, 0.22);
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: rgba(15, 23, 42, 0.98);
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      color: #dbeafe;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    tbody td {{
      padding: 11px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
      vertical-align: top;
    }}
    tbody tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
    .metric-cell {{
      font-weight: 600;
      color: #eff6ff;
      white-space: nowrap;
    }}
    .num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .bad-cell {{ color: #fecdd3; }}
    .good-cell {{ color: #a7f3d0; }}
    .hot {{
      background: rgba(244, 63, 94, 0.14);
    }}
    .cold {{
      background: rgba(45, 212, 191, 0.10);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .badge.bad {{ background: rgba(244, 63, 94, 0.14); color: #fda4af; }}
    .badge.good {{ background: rgba(45, 212, 191, 0.13); color: #99f6e4; }}
    .badge.warn {{ background: rgba(245, 158, 11, 0.15); color: #fcd34d; }}
    .insight {{
      grid-column: span 12;
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }}
    .insight-panel {{
      grid-column: span 6;
      padding: 18px;
    }}
    .outlier-list {{
      display: grid;
      gap: 10px;
    }}
    .outlier-item {{
      display: grid;
      grid-template-columns: 110px 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 11px 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .outlier-step {{
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .muted {{ color: var(--muted); }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }}
    .pill-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .pill {{
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 12px;
      color: #dbeafe;
    }}
    @media (max-width: 1100px) {{
      .summary, .focus-card, .insight-panel {{ grid-column: span 12; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
      .detail-hero {{ grid-template-columns: 1fr; }}
      .heatmap {{ grid-template-columns: 140px repeat(4, minmax(72px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Station timing comparison</div>
      <h1>Bad host vs good host</h1>
      <div class="subtitle">
        This report compares the two benchmark summaries side by side and highlights where the bad machine
        spends more time. It is intentionally focused on the host-side follower path, pacing jitter, and
        component span, because that is the part that diverged most strongly in the standalone run.
      </div>
      <div class="pill-row">
        <span class="pill">{html.escape(str(bad_path))}</span>
        <span class="pill">{html.escape(str(good_path))}</span>
      </div>
    </section>

    <section class="grid">
      <article class="card summary">
        <h2>{html.escape(bad_card["label"])} summary</h2>
        <div class="summary-meta">
          {_render_summary_kv(bad_card)}
        </div>
      </article>
      <article class="card summary">
        <h2>{html.escape(good_card["label"])} summary</h2>
        <div class="summary-meta">
          {_render_summary_kv(good_card)}
        </div>
      </article>

      <article class="card section" style="grid-column: span 12;">
        <h2>Focus metrics</h2>
        <div class="focus-grid" id="focus-grid"></div>
      </article>

      <article class="card section" style="grid-column: span 12;">
        <h2>Metric map</h2>
        <div class="toolbar">
          <div class="toolbar-label">Scatter stat</div>
          <div class="segmented" id="stat-toggle"></div>
        </div>
        <div class="chart-grid">
          <div class="chart-panel">
            <h3>Bad vs good scatter</h3>
            <div class="chart-caption">
              Each point is a metric. The diagonal means parity. Points farther above the diagonal are worse on the bad host.
              Hover for values. Click a point to inspect that metric in detail.
            </div>
            <svg id="metric-scatter" class="chart-svg" viewBox="0 0 760 320" preserveAspectRatio="none"></svg>
          </div>
          <div class="chart-panel">
            <h3>Metric drilldown</h3>
            <div class="chart-caption">
              Selected metric detail across mean, p99, max, and count of samples above 10 ms.
            </div>
            <div id="metric-detail" class="detail-grid"></div>
          </div>
        </div>
      </article>

      <article class="card section" style="grid-column: span 12;">
        <h2>Ratio heatmap</h2>
        <div class="chart-caption">
          Click any cell to select that metric. Warmer cells mean the bad host is slower by a larger factor.
        </div>
        <div id="ratio-heatmap" class="heatmap"></div>
      </article>

      <article class="card section" style="grid-column: span 12;">
        <h2>Outlier explorer</h2>
        <div class="toolbar">
          <div class="toolbar-label">Host</div>
          <div class="segmented" id="outlier-host-toggle"></div>
          <div class="toolbar-label">Metric</div>
          <div class="segmented" id="outlier-metric-toggle"></div>
        </div>
        <div class="chart-grid">
          <div class="chart-panel">
            <h3>Outlier strip plot</h3>
            <div class="chart-caption">
              Top outlier events for the selected host and metric. Hover a point to inspect its step and supporting timing buckets.
            </div>
            <svg id="outlier-chart" class="chart-svg" viewBox="0 0 760 320" preserveAspectRatio="none"></svg>
          </div>
          <div class="chart-panel">
            <h3>Selected outlier detail</h3>
            <div class="chart-caption">
              Breakdown of the currently selected outlier event using the fields saved in the benchmark summaries.
            </div>
            <div id="outlier-detail" class="detail-grid"></div>
          </div>
        </div>
      </article>

      <article class="card section" style="grid-column: span 12;">
        <h2>Reading guide</h2>
        <div class="footer">
          A ratio above <strong>1.0x</strong> means the bad host is slower. For this problem, the most important
          columns are <code>env_obs_phase_ms</code>, <code>left_follower_wait_ms</code>,
          <code>right_follower_wait_ms</code>, <code>obs_component_span_ms</code>, and
          <code>pacing_overshoot_ms</code>. Those are the buckets that separated the machines most clearly.
        </div>
        <details class="table-details">
          <summary>Open full metric table</summary>
          <div class="table-wrap">
            <table id="metrics-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>{html.escape(bad_label)} mean</th>
                  <th>{html.escape(good_label)} mean</th>
                  <th>Mean ratio</th>
                  <th>{html.escape(bad_label)} p99</th>
                  <th>{html.escape(good_label)} p99</th>
                  <th>p99 ratio</th>
                  <th>{html.escape(bad_label)} max</th>
                  <th>{html.escape(good_label)} max</th>
                  <th>Max ratio</th>
                  <th>{html.escape(bad_label)} >10ms</th>
                  <th>{html.escape(good_label)} >10ms</th>
                </tr>
              </thead>
              <tbody id="metrics-body"></tbody>
            </table>
          </div>
        </details>
      </article>
    </section>
  </div>
  <div id="tooltip" class="tooltip"></div>

  <script id="timing-data" type="application/json">{payload_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('timing-data').textContent);
    const TOOLTIP = document.getElementById('tooltip');
    const STAT_OPTIONS = [
      {{ key: 'mean', label: 'Mean' }},
      {{ key: 'p99', label: 'p99' }},
      {{ key: 'max', label: 'Max' }},
      {{ key: 'gt10', label: '>10ms' }},
    ];
    const OUTLIER_PREFERRED = ['env_obs_phase_ms', 'obs_component_span_ms', 'pacing_overshoot_ms', 'loop_total_ms'];
    const STATE = {{
      stat: 'p99',
      selectedMetric: (DATA.focus[0] && DATA.focus[0].metric) || (DATA.metrics[0] && DATA.metrics[0].metric),
      outlierHost: 'bad',
      outlierMetric: null,
      outlierIndex: 0,
    }};

    function fmtMs(v) {{
      return Number.isFinite(v) ? `${{v.toFixed(3)}} ms` : 'n/a';
    }}

    function fmtRatio(v) {{
      if (!Number.isFinite(v)) return 'n/a';
      return `${{v.toFixed(2)}}x`;
    }}

    function fmtPct(v) {{
      if (!Number.isFinite(v)) return 'n/a';
      const sign = v > 0 ? '+' : '';
      return `${{sign}}${{v.toFixed(1)}}%`;
    }}

    function fmtCount(v) {{
      return Number.isFinite(v) ? `${{Math.round(v)}}` : 'n/a';
    }}

    function esc(text) {{
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }}

    function valueFor(row, host, stat) {{
      return row && row[host] ? row[host][stat] : NaN;
    }}

    function ratioFor(row, stat) {{
      const bad = valueFor(row, 'bad', stat);
      const good = valueFor(row, 'good', stat);
      if (!Number.isFinite(bad) || !Number.isFinite(good)) return NaN;
      if (good === 0) return bad > 0 ? Infinity : 1;
      return bad / good;
    }}

    function metricByName(name) {{
      return DATA.metrics.find(row => row.metric === name) || DATA.focus[0] || DATA.metrics[0];
    }}

    function currentOutlierGroups() {{
      return STATE.outlierHost === 'bad' ? DATA.top_bad : DATA.top_good;
    }}

    function outlierKeysForHost(host) {{
      const groups = host === 'bad' ? DATA.top_bad : DATA.top_good;
      return OUTLIER_PREFERRED.filter(k => groups[k] && groups[k].length)
        .concat(Object.keys(groups).filter(k => !OUTLIER_PREFERRED.includes(k) && groups[k] && groups[k].length));
    }}

    function ensureOutlierMetric() {{
      const keys = outlierKeysForHost(STATE.outlierHost);
      if (!keys.length) {{
        STATE.outlierMetric = null;
        STATE.outlierIndex = 0;
        return;
      }}
      if (!STATE.outlierMetric || !keys.includes(STATE.outlierMetric)) {{
        STATE.outlierMetric = keys[0];
        STATE.outlierIndex = 0;
      }}
      const rows = currentOutlierGroups()[STATE.outlierMetric] || [];
      if (STATE.outlierIndex >= rows.length) {{
        STATE.outlierIndex = 0;
      }}
    }}

    function metricStatus(row) {{
      const ratio = Number.isFinite(row.ratio_p99) ? row.ratio_p99 : 0;
      if (ratio >= 2.0) return 'bad';
      if (ratio <= 1.1) return 'good';
      return 'warn';
    }}

    function colorForRatio(ratio) {{
      if (!Number.isFinite(ratio)) return 'rgba(148,163,184,0.35)';
      if (ratio >= 8) return 'rgba(244,63,94,0.95)';
      if (ratio >= 3) return 'rgba(251,113,133,0.88)';
      if (ratio >= 1.4) return 'rgba(245,158,11,0.88)';
      if (ratio >= 1.0) return 'rgba(96,165,250,0.82)';
      return 'rgba(45,212,191,0.88)';
    }}

    function heatColor(ratio) {{
      if (!Number.isFinite(ratio)) return 'rgba(148, 163, 184, 0.08)';
      const capped = Math.max(0, Math.min((Math.log2(Math.max(ratio, 1e-6)) + 1) / 4, 1));
      const red = 30 + Math.round(220 * capped);
      const green = 56 + Math.round(100 * (1 - capped));
      const blue = 98 + Math.round(60 * (1 - capped));
      return `rgba(${{red}}, ${{green}}, ${{blue}}, 0.30)`;
    }}

    function showTooltip(evt, htmlText) {{
      TOOLTIP.innerHTML = htmlText;
      TOOLTIP.classList.add('visible');
      TOOLTIP.style.left = `${{evt.clientX + 14}}px`;
      TOOLTIP.style.top = `${{evt.clientY + 14}}px`;
    }}

    function moveTooltip(evt) {{
      TOOLTIP.style.left = `${{evt.clientX + 14}}px`;
      TOOLTIP.style.top = `${{evt.clientY + 14}}px`;
    }}

    function hideTooltip() {{
      TOOLTIP.classList.remove('visible');
    }}

    function lineTicks(maxValue, count) {{
      const ticks = [];
      for (let i = 0; i <= count; i += 1) {{
        ticks.push((maxValue * i) / count);
      }}
      return ticks;
    }}

    function renderToggle(targetId, options, activeKey, onClick) {{
      const el = document.getElementById(targetId);
      el.innerHTML = options.map(opt => `
        <button class="${{opt.key === activeKey ? 'active' : ''}}" data-key="${{opt.key}}">
          ${{opt.label}}
        </button>
      `).join('');
      el.querySelectorAll('button').forEach(button => {{
        button.addEventListener('click', () => onClick(button.dataset.key));
      }});
    }}

    function renderFocus() {{
      const el = document.getElementById('focus-grid');
      const rows = DATA.focus;
      const maxBad = Math.max(...rows.map(r => r.bad.p99 || 0), 1e-6);
      const maxGood = Math.max(...rows.map(r => r.good.p99 || 0), 1e-6);
      el.innerHTML = rows.map(row => {{
        const status = metricStatus(row);
        const badWidth = Math.max(4, (row.bad.p99 / maxBad) * 100);
        const goodWidth = Math.max(4, (row.good.p99 / maxGood) * 100);
        return `
          <div class="focus-card">
            <div class="metric-name">${{row.metric}}</div>
            <div class="bar-row">
              <div class="bar-label">bad</div>
              <div class="bar-track"><div class="bar-fill bad" style="width:${{badWidth}}%"></div></div>
              <div class="num">${{fmtMs(row.bad.p99)}}</div>
            </div>
            <div class="bar-row">
              <div class="bar-label">good</div>
              <div class="bar-track"><div class="bar-fill good" style="width:${{goodWidth}}%"></div></div>
              <div class="num">${{fmtMs(row.good.p99)}}</div>
            </div>
            <div class="delta ${{status}}">
              p99 ratio <strong>${{fmtRatio(row.ratio_p99)}}</strong>
              <span class="muted">| delta ${{fmtPct(row.pct_p99)}}</span>
            </div>
            </div>`;
      }}).join('');
    }}

    function renderMetricMap() {{
      renderToggle('stat-toggle', STAT_OPTIONS, STATE.stat, key => {{
        STATE.stat = key;
        renderMetricMap();
        renderHeatmap();
      }});

      const svg = document.getElementById('metric-scatter');
      const rows = DATA.metrics.filter(row => Number.isFinite(valueFor(row, 'bad', STATE.stat)) && Number.isFinite(valueFor(row, 'good', STATE.stat)));
      const width = 760;
      const height = 320;
      const pad = {{ left: 64, right: 26, top: 18, bottom: 46 }};
      const maxValue = Math.max(
        1e-6,
        ...rows.flatMap(row => [valueFor(row, 'bad', STATE.stat), valueFor(row, 'good', STATE.stat)])
      ) * 1.08;
      const x = value => pad.left + (value / maxValue) * (width - pad.left - pad.right);
      const y = value => height - pad.bottom - (value / maxValue) * (height - pad.top - pad.bottom);
      const ticks = lineTicks(maxValue, 4);

      svg.innerHTML = `
        ${{ticks.map(t => `
          <line class="grid-line" x1="${{x(t)}}" y1="${{pad.top}}" x2="${{x(t)}}" y2="${{height - pad.bottom}}"></line>
          <line class="grid-line" x1="${{pad.left}}" y1="${{y(t)}}" x2="${{width - pad.right}}" y2="${{y(t)}}"></line>
          <text class="tick-label" x="${{x(t)}}" y="${{height - 20}}" text-anchor="middle">${{STATE.stat === 'gt10' ? Math.round(t) : t.toFixed(1)}}</text>
          <text class="tick-label" x="52" y="${{y(t) + 4}}" text-anchor="end">${{STATE.stat === 'gt10' ? Math.round(t) : t.toFixed(1)}}</text>
        `).join('')}}
        <line class="diag-line" x1="${{x(0)}}" y1="${{y(0)}}" x2="${{x(maxValue)}}" y2="${{y(maxValue)}}"></line>
        <text class="axis-label" x="${{width / 2}}" y="${{height - 6}}" text-anchor="middle">good host ${{STATE.stat}}</text>
        <text class="axis-label" x="16" y="${{height / 2}}" text-anchor="middle" transform="rotate(-90 16 ${{height / 2}})">bad host ${{STATE.stat}}</text>
        ${{rows.map((row, idx) => {{
          const bad = valueFor(row, 'bad', STATE.stat);
          const good = valueFor(row, 'good', STATE.stat);
          const ratio = ratioFor(row, STATE.stat);
          const selected = row.metric === STATE.selectedMetric;
          const radius = selected ? 8.5 : Math.max(4.5, Math.min(11, 4 + Math.log2(Math.max(ratio, 1)) * 2.2));
          return `
            <circle
              class="metric-point ${{selected ? 'selected' : ''}}"
              data-index="${{idx}}"
              cx="${{x(good)}}"
              cy="${{y(bad)}}"
              r="${{radius}}"
              fill="${{colorForRatio(ratio)}}"
              stroke="${{selected ? '#ffffff' : 'rgba(255,255,255,0.16)'}}"
              stroke-width="${{selected ? 2.2 : 1.1}}"
            ></circle>
          `;
        }}).join('')}}
      `;

      svg.querySelectorAll('.metric-point').forEach(node => {{
        const row = rows[Number(node.dataset.index)];
        node.addEventListener('mouseenter', evt => showTooltip(evt, `
          <strong>${{esc(row.metric)}}</strong><br>
          bad: ${{STATE.stat === 'gt10' ? fmtCount(valueFor(row, 'bad', STATE.stat)) : fmtMs(valueFor(row, 'bad', STATE.stat))}}<br>
          good: ${{STATE.stat === 'gt10' ? fmtCount(valueFor(row, 'good', STATE.stat)) : fmtMs(valueFor(row, 'good', STATE.stat))}}<br>
          ratio: ${{fmtRatio(ratioFor(row, STATE.stat))}}
        `));
        node.addEventListener('mousemove', moveTooltip);
        node.addEventListener('mouseleave', hideTooltip);
        node.addEventListener('click', () => {{
          STATE.selectedMetric = row.metric;
          renderMetricMap();
          renderHeatmap();
          renderMetricDetail();
        }});
      }});

      renderMetricDetail();
    }}

    function renderMetricDetail() {{
      const row = metricByName(STATE.selectedMetric);
      const el = document.getElementById('metric-detail');
      if (!row) {{
        el.innerHTML = '<div class="muted">No metric selected.</div>';
        return;
      }}
      const metrics = ['mean', 'p99', 'max', 'gt10'];
      const maxValue = Math.max(1e-6, ...metrics.flatMap(stat => [valueFor(row, 'bad', stat), valueFor(row, 'good', stat)].filter(Number.isFinite)));
      el.innerHTML = `
        <div class="detail-hero">
          <div class="detail-stat">
            <div class="k">Selected metric</div>
            <div class="v" style="font-size:16px;">${{esc(row.metric)}}</div>
          </div>
          <div class="detail-stat">
            <div class="k">Selected stat ratio</div>
            <div class="v">${{fmtRatio(ratioFor(row, STATE.stat))}}</div>
          </div>
          <div class="detail-stat">
            <div class="k">Selected stat delta</div>
            <div class="v">${{fmtPct(row[`pct_${{STATE.stat}}`] ?? ((valueFor(row, 'bad', STATE.stat) - valueFor(row, 'good', STATE.stat)) / Math.max(valueFor(row, 'good', STATE.stat), 1e-9) * 100))}}</div>
          </div>
        </div>
        <div class="mini-bars">
          ${{metrics.map(stat => {{
            const bad = valueFor(row, 'bad', stat);
            const good = valueFor(row, 'good', stat);
            const badWidth = Math.max(4, (bad / maxValue) * 100);
            const goodWidth = Math.max(4, (good / maxValue) * 100);
            const badText = stat === 'gt10' ? fmtCount(bad) : fmtMs(bad);
            const goodText = stat === 'gt10' ? fmtCount(good) : fmtMs(good);
            return `
              <div class="mini-row">
                <div class="bar-label">${{stat}}</div>
                <div class="mini-track">
                  <div class="mini-fill bad" style="width:${{badWidth}}%"></div>
                </div>
                <div class="num bad-cell">${{badText}}</div>
                <div class="num">${{fmtRatio(ratioFor(row, stat))}}</div>
              </div>
              <div class="mini-row" style="margin-top:-2px;">
                <div class="bar-label muted">good</div>
                <div class="mini-track">
                  <div class="mini-fill good" style="width:${{goodWidth}}%"></div>
                </div>
                <div class="num good-cell">${{goodText}}</div>
                <div class="num muted">baseline</div>
              </div>
            `;
          }}).join('')}}
        </div>
      `;
    }}

    function renderHeatmap() {{
      const el = document.getElementById('ratio-heatmap');
      const stats = ['mean', 'p99', 'max', 'gt10'];
      const sortedRows = DATA.metrics.slice().sort((a, b) => {{
        const ar = ratioFor(a, STATE.stat);
        const br = ratioFor(b, STATE.stat);
        return (Number.isFinite(br) ? br : -1) - (Number.isFinite(ar) ? ar : -1);
      }});
      el.innerHTML = `
        <div></div>
        ${{stats.map(stat => `<div class="heatmap-head">${{stat}}</div>`).join('')}}
        ${{sortedRows.map(row => `
          <div class="heatmap-metric">${{esc(row.metric)}}</div>
          ${{stats.map(stat => {{
            const ratio = ratioFor(row, stat);
            const selected = row.metric === STATE.selectedMetric && stat === STATE.stat;
            return `
              <div
                class="heat-cell ${{selected ? 'selected' : ''}}"
                data-metric="${{row.metric}}"
                data-stat="${{stat}}"
                style="background:${{heatColor(ratio)}}"
              >
                ${{fmtRatio(ratio)}}
              </div>
            `;
          }}).join('')}}
        `).join('')}}
      `;
      el.querySelectorAll('.heat-cell').forEach(cell => {{
        const row = metricByName(cell.dataset.metric);
        const stat = cell.dataset.stat;
        cell.addEventListener('mouseenter', evt => showTooltip(evt, `
          <strong>${{esc(row.metric)}}</strong><br>
          stat: ${{esc(stat)}}<br>
          bad: ${{stat === 'gt10' ? fmtCount(valueFor(row, 'bad', stat)) : fmtMs(valueFor(row, 'bad', stat))}}<br>
          good: ${{stat === 'gt10' ? fmtCount(valueFor(row, 'good', stat)) : fmtMs(valueFor(row, 'good', stat))}}<br>
          ratio: ${{fmtRatio(ratioFor(row, stat))}}
        `));
        cell.addEventListener('mousemove', moveTooltip);
        cell.addEventListener('mouseleave', hideTooltip);
        cell.addEventListener('click', () => {{
          STATE.selectedMetric = cell.dataset.metric;
          STATE.stat = stat;
          renderMetricMap();
          renderHeatmap();
        }});
      }});
    }}

    function renderMetricsTable() {{
      const tbody = document.getElementById('metrics-body');
      const rows = DATA.metrics.slice().sort((a, b) => {{
        const ar = Number.isFinite(a.ratio_p99) ? a.ratio_p99 : -1;
        const br = Number.isFinite(b.ratio_p99) ? b.ratio_p99 : -1;
        return br - ar;
      }});
      tbody.innerHTML = rows.map(row => {{
        const status = metricStatus(row);
        return `
          <tr>
            <td class="metric-cell">${{row.metric}}</td>
            <td class="num bad-cell">${{fmtMs(row.bad.mean)}}</td>
            <td class="num good-cell">${{fmtMs(row.good.mean)}}</td>
            <td class="num ${{status}}">${{fmtRatio(row.bad.mean / row.good.mean)}}</td>
            <td class="num bad-cell">${{fmtMs(row.bad.p99)}}</td>
            <td class="num good-cell">${{fmtMs(row.good.p99)}}</td>
            <td class="num ${{status}}">${{fmtRatio(row.ratio_p99)}}</td>
            <td class="num bad-cell">${{fmtMs(row.bad.max)}}</td>
            <td class="num good-cell">${{fmtMs(row.good.max)}}</td>
            <td class="num ${{status}}">${{fmtRatio(row.ratio_max)}}</td>
            <td class="num bad-cell">${{Number.isFinite(row.bad.gt10) ? row.bad.gt10 : 'n/a'}}</td>
            <td class="num good-cell">${{Number.isFinite(row.good.gt10) ? row.good.gt10 : 'n/a'}}</td>
          </tr>`;
      }}).join('');
    }}

    function renderOutlierExplorer() {{
      ensureOutlierMetric();
      renderToggle('outlier-host-toggle', [
        {{ key: 'bad', label: DATA.bad_card.label || 'bad host' }},
        {{ key: 'good', label: DATA.good_card.label || 'good host' }},
      ], STATE.outlierHost, key => {{
        STATE.outlierHost = key;
        ensureOutlierMetric();
        renderOutlierExplorer();
      }});

      const metricOptions = outlierKeysForHost(STATE.outlierHost).map(key => ({{
        key,
        label: key.replaceAll('_ms', '').replaceAll('_', ' '),
      }}));
      renderToggle('outlier-metric-toggle', metricOptions, STATE.outlierMetric, key => {{
        STATE.outlierMetric = key;
        STATE.outlierIndex = 0;
        renderOutlierExplorer();
      }});

      const svg = document.getElementById('outlier-chart');
      const rows = (currentOutlierGroups()[STATE.outlierMetric] || []).slice();
      const width = 760;
      const height = 320;
      const pad = {{ left: 64, right: 26, top: 18, bottom: 46 }};
      if (!rows.length) {{
        svg.innerHTML = `<text class="tick-label" x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle">No outliers for this host/metric.</text>`;
        document.getElementById('outlier-detail').innerHTML = '<div class="muted">No outliers recorded.</div>';
        return;
      }}
      const xMin = Math.min(...rows.map(r => r.step));
      const xMax = Math.max(...rows.map(r => r.step));
      const yMax = Math.max(...rows.map(r => Number(r[STATE.outlierMetric] || 0)), 1e-6) * 1.08;
      const x = value => pad.left + (((value - xMin) / Math.max(xMax - xMin, 1)) * (width - pad.left - pad.right));
      const y = value => height - pad.bottom - ((value / yMax) * (height - pad.top - pad.bottom));
      const ticks = lineTicks(yMax, 4);
      svg.innerHTML = `
        ${{ticks.map(t => `
          <line class="grid-line" x1="${{pad.left}}" y1="${{y(t)}}" x2="${{width - pad.right}}" y2="${{y(t)}}"></line>
          <text class="tick-label" x="52" y="${{y(t) + 4}}" text-anchor="end">${{t.toFixed(1)}}</text>
        `).join('')}}
        <text class="axis-label" x="${{width / 2}}" y="${{height - 6}}" text-anchor="middle">benchmark step</text>
        <text class="axis-label" x="16" y="${{height / 2}}" text-anchor="middle" transform="rotate(-90 16 ${{height / 2}})">${{STATE.outlierMetric}}</text>
        ${{rows.map((row, idx) => {{
          const selected = idx === STATE.outlierIndex;
          return `
            <circle
              class="metric-point ${{selected ? 'selected' : ''}}"
              data-index="${{idx}}"
              cx="${{x(row.step)}}"
              cy="${{y(Number(row[STATE.outlierMetric] || 0))}}"
              r="${{selected ? 8.5 : 6.2}}"
              fill="${{STATE.outlierHost === 'bad' ? 'rgba(251,113,133,0.92)' : 'rgba(45,212,191,0.92)'}}"
              stroke="${{selected ? '#ffffff' : 'rgba(255,255,255,0.14)'}}"
              stroke-width="${{selected ? 2.2 : 1.0}}"
            ></circle>
          `;
        }}).join('')}}
      `;
      svg.querySelectorAll('.metric-point').forEach(node => {{
        const row = rows[Number(node.dataset.index)];
        node.addEventListener('mouseenter', evt => showTooltip(evt, `
          <strong>step ${{row.step}}</strong><br>
          metric: ${{esc(STATE.outlierMetric)}}<br>
          value: ${{fmtMs(Number(row[STATE.outlierMetric] || 0))}}<br>
          loop: ${{fmtMs(Number(row.loop_total_ms || 0))}}<br>
          env: ${{fmtMs(Number(row.env_obs_phase_ms || 0))}}<br>
          pace: ${{fmtMs(Number(row.pacing_overshoot_ms || 0))}}
        `));
        node.addEventListener('mousemove', moveTooltip);
        node.addEventListener('mouseleave', hideTooltip);
        node.addEventListener('click', () => {{
          STATE.outlierIndex = Number(node.dataset.index);
          renderOutlierExplorer();
        }});
      }});

      renderOutlierDetail(rows[STATE.outlierIndex], STATE.outlierMetric);
    }}

    function renderOutlierDetail(row, metric) {{
      const el = document.getElementById('outlier-detail');
      if (!row) {{
        el.innerHTML = '<div class="muted">No outlier selected.</div>';
        return;
      }}
      const detailMetrics = [
        metric,
        'loop_total_ms',
        'env_obs_phase_ms',
        'pacing_overshoot_ms',
        'left_follower_wait_ms',
        'right_follower_wait_ms',
        'obs_component_span_ms',
        'top_camera_get_image_ms',
        'left_camera_get_image_ms',
        'right_camera_get_image_ms',
      ].filter((name, idx, arr) => arr.indexOf(name) === idx && Number.isFinite(Number(row[name] || NaN)));
      const maxValue = Math.max(1e-6, ...detailMetrics.map(name => Number(row[name] || 0)));
      el.innerHTML = `
        <div class="detail-hero">
          <div class="detail-stat">
            <div class="k">Step</div>
            <div class="v">${{fmtCount(row.step)}}</div>
          </div>
          <div class="detail-stat">
            <div class="k">Wall time</div>
            <div class="v" style="font-size:16px;">${{esc(row.wall_time || 'n/a')}}</div>
          </div>
          <div class="detail-stat">
            <div class="k">Selected metric</div>
            <div class="v">${{fmtMs(Number(row[metric] || 0))}}</div>
          </div>
        </div>
        <div class="mini-bars">
          ${{detailMetrics.map(name => {{
            const value = Number(row[name] || 0);
            const width = Math.max(4, (value / maxValue) * 100);
            return `
              <div class="mini-row">
                <div class="bar-label">${{esc(name.replaceAll('_ms', '').replaceAll('_', ' '))}}</div>
                <div class="mini-track"><div class="mini-fill ${{name === metric ? 'bad' : 'good'}}" style="width:${{width}}%"></div></div>
                <div class="num">${{fmtMs(value)}}</div>
                <div class="num muted">${{((value / maxValue) * 100).toFixed(0)}}%</div>
              </div>
            `;
          }}).join('')}}
        </div>
      `;
    }}

    renderFocus();
    renderMetricMap();
    renderHeatmap();
    renderMetricsTable();
    renderOutlierExplorer();
  </script>
</body>
</html>"""


def _render_summary_kv(card: dict[str, Any]) -> str:
    fields = [
        ("Hostname", card.get("hostname")),
        ("Station", card.get("station_key")),
        ("Platform", card.get("platform")),
        ("Python", card.get("python_version")),
        ("Target Hz", card.get("target_hz")),
        ("Iterations", card.get("iterations_recorded")),
        ("Leaders", card.get("include_leaders")),
        ("Followers", card.get("include_followers")),
        ("Cameras", card.get("include_cameras")),
        ("Created", card.get("created_at")),
    ]
    return "\n".join(
        f'<div class="kv"><div class="k">{html.escape(label)}</div><div class="v">{html.escape("n/a" if value is None else str(value))}</div></div>'
        for label, value in fields
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two station timing summary JSON files")
    parser.add_argument("bad_summary", type=Path, help="Path to the bad-host summary JSON")
    parser.add_argument("good_summary", type=Path, help="Path to the good-host summary JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("station_timing_compare.html"),
        help="Output HTML file path",
    )
    args = parser.parse_args()

    bad = _load_json(args.bad_summary)
    good = _load_json(args.good_summary)
    html_text = build_report(bad, good, args.bad_summary, args.good_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote comparison report to {args.output}")


if __name__ == "__main__":
    main()
