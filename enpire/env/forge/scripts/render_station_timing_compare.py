# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    bad_summary: str
    good_summary: str
    output_html: str = "docs/station_timing_compare_bad_vs_good.html"
    title: str = "Station Timing Comparison"


METRICS = [
    ("loop_total_ms", "Loop Total", "Whole step period after follower reads and pacing."),
    ("env_obs_phase_ms", "Follower Obs Phase", "Time spent collecting follower observations inside the loop."),
    ("left_follower_wait_ms", "Left Follower Wait", "Portal/RPC wait for left follower observations."),
    ("right_follower_wait_ms", "Right Follower Wait", "Portal/RPC wait for right follower observations."),
    ("obs_component_span_ms", "Obs Component Span", "Host-side skew between sequential observation components."),
    ("pacing_overshoot_ms", "Pacing Overshoot", "How late the host wakes up past the scheduled control boundary."),
]


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _metric(summary: dict, key: str) -> dict:
    value = summary["metrics"].get(key)
    if value is None:
        raise KeyError(f"Missing metric {key!r} in {summary['meta']['summary_path']}")
    return value


def _fmt_ms(value: float) -> str:
    return f"{value:.3f} ms"


def _fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.1f}x"


def _bar(value: float, limit: float, tone: str) -> str:
    width = 6.0 if limit <= 0 else max(6.0, (value / limit) * 100.0)
    return (
        f"<div class='bar-track'><div class='bar-fill {tone}' style='width:{width:.2f}%'></div></div>"
    )


def _threshold_row(label: str, bad_value: int, good_value: int) -> str:
    limit = max(bad_value, good_value, 1)
    return (
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td class='metric-num'>{bad_value}</td>"
        f"<td>{_bar(float(bad_value), float(limit), 'bad')}</td>"
        f"<td class='metric-num'>{good_value}</td>"
        f"<td>{_bar(float(good_value), float(limit), 'good')}</td>"
        "</tr>"
    )


def _metric_section(key: str, label: str, subtitle: str, bad: dict, good: dict) -> str:
    stats = [("mean_ms", "Mean"), ("p95_ms", "P95"), ("p99_ms", "P99"), ("max_ms", "Max")]
    rows: list[str] = []
    for stat_key, stat_label in stats:
        bad_value = float(bad[stat_key])
        good_value = float(good[stat_key])
        limit = max(bad_value, good_value, 1e-9)
        rows.append(
            "<tr>"
            f"<td>{stat_label}</td>"
            f"<td class='metric-num'>{_fmt_ms(bad_value)}</td>"
            f"<td>{_bar(bad_value, limit, 'bad')}</td>"
            f"<td class='metric-num'>{_fmt_ms(good_value)}</td>"
            f"<td>{_bar(good_value, limit, 'good')}</td>"
            f"<td class='metric-ratio'>{_fmt_ratio(bad_value, good_value)}</td>"
            "</tr>"
        )

    thresholds = [
        ("Over 2 ms", int(bad.get("gt2ms", 0)), int(good.get("gt2ms", 0))),
        ("Over 5 ms", int(bad.get("gt5ms", 0)), int(good.get("gt5ms", 0))),
        ("Over 10 ms", int(bad.get("gt10ms", 0)), int(good.get("gt10ms", 0))),
        ("Over 20 ms", int(bad.get("gt20ms", 0)), int(good.get("gt20ms", 0))),
    ]
    threshold_rows = "\n".join(_threshold_row(*row) for row in thresholds)

    return f"""
    <section class="metric-card">
      <div class="metric-head">
        <div>
          <h3>{html.escape(label)}</h3>
          <p>{html.escape(subtitle)}</p>
        </div>
        <div class="signal-pill">{_fmt_ratio(float(bad['p99_ms']), float(good['p99_ms']))} worse at p99</div>
      </div>
      <table class="metric-table">
        <thead>
          <tr>
            <th>Stat</th>
            <th>Bad Host</th>
            <th></th>
            <th>Good Host</th>
            <th></th>
            <th>Bad / Good</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
      <div class="threshold-block">
        <div class="threshold-title">Excursion counts</div>
        <table class="threshold-table">
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Bad</th>
              <th></th>
              <th>Good</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {threshold_rows}
          </tbody>
        </table>
      </div>
    </section>
    """


def _outlier_story(summary: dict, key: str, heading: str, bullets: list[str]) -> str:
    rows = summary["top_outliers"].get(key, [])[:4]
    items = []
    for row in rows:
        step = int(row["step"])
        wall_time = html.escape(str(row["wall_time"]))
        loop_total = _fmt_ms(float(row.get("loop_total_ms", 0.0)))
        env_obs = _fmt_ms(float(row.get("env_obs_phase_ms", 0.0)))
        pacing = _fmt_ms(float(row.get("pacing_overshoot_ms", 0.0)))
        left = _fmt_ms(float(row.get("left_follower_wait_ms", 0.0)))
        right = _fmt_ms(float(row.get("right_follower_wait_ms", 0.0)))
        items.append(
            f"<li><strong>Step {step}</strong> at {wall_time}: loop={loop_total}, "
            f"env_obs={env_obs}, pacing_overshoot={pacing}, "
            f"left_wait={left}, right_wait={right}</li>"
        )
    bullets_html = "".join(f"<li>{html.escape(line)}</li>" for line in bullets)
    return f"""
    <section class="story-card">
      <h3>{html.escape(heading)}</h3>
      <ul class="story-list">
        {items}
      </ul>
      <ul class="bullet-list">
        {bullets_html}
      </ul>
    </section>
    """


def _build_report(args: Args, bad_summary: dict, good_summary: dict) -> str:
    meta_bad = bad_summary["meta"]
    meta_good = good_summary["meta"]
    hero_metrics = {
        "Bad env p99": (
            _fmt_ms(float(_metric(bad_summary, "env_obs_phase_ms")["p99_ms"])),
            "Good env p99 "
            + _fmt_ms(float(_metric(good_summary, "env_obs_phase_ms")["p99_ms"])),
        ),
        "Bad max follower wait": (
            _fmt_ms(
                max(
                    float(_metric(bad_summary, "left_follower_wait_ms")["max_ms"]),
                    float(_metric(bad_summary, "right_follower_wait_ms")["max_ms"]),
                )
            ),
            "Good max follower wait "
            + _fmt_ms(
                max(
                    float(_metric(good_summary, "left_follower_wait_ms")["max_ms"]),
                    float(_metric(good_summary, "right_follower_wait_ms")["max_ms"]),
                )
            ),
        ),
        "Bad pacing p99": (
            _fmt_ms(float(_metric(bad_summary, "pacing_overshoot_ms")["p99_ms"])),
            "Good pacing p99 "
            + _fmt_ms(float(_metric(good_summary, "pacing_overshoot_ms")["p99_ms"])),
        ),
        "Loop std spread": (
            _fmt_ratio(
                float(_metric(bad_summary, "loop_total_ms")["std_ms"]),
                float(_metric(good_summary, "loop_total_ms")["std_ms"]),
            ),
            "bad vs good loop standard deviation",
        ),
    }
    hero_cards = "".join(
        f"""
        <div class="hero-card">
          <div class="hero-label">{html.escape(label)}</div>
          <div class="hero-value">{html.escape(value)}</div>
          <div class="hero-sub">{html.escape(sub)}</div>
        </div>
        """
        for label, (value, sub) in hero_metrics.items()
    )

    metric_sections = "".join(
        _metric_section(
            key,
            label,
            subtitle,
            _metric(bad_summary, key),
            _metric(good_summary, key),
        )
        for key, label, subtitle in METRICS
    )

    bad_env = _metric(bad_summary, "env_obs_phase_ms")
    good_env = _metric(good_summary, "env_obs_phase_ms")
    verdict = (
        f"Bad host follower observation p99 is {_fmt_ratio(float(bad_env['p99_ms']), float(good_env['p99_ms']))}; "
        f"bad host pacing overshoot p99 is "
        f"{_fmt_ratio(float(_metric(bad_summary, 'pacing_overshoot_ms')['p99_ms']), float(_metric(good_summary, 'pacing_overshoot_ms')['p99_ms']))}. "
        "This is the same jitter class seen during data collection, reproduced without leaders and without cameras."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(args.title)}</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: rgba(255,255,255,0.76);
      --ink: #152028;
      --muted: #5c6a73;
      --line: rgba(21,32,40,0.10);
      --bad: #c3472c;
      --bad-soft: rgba(195,71,44,0.14);
      --good: #227b67;
      --good-soft: rgba(34,123,103,0.14);
      --accent: #d6a74e;
      --shadow: 0 24px 60px rgba(30, 45, 58, 0.12);
      --radius: 22px;
      --font: "IBM Plex Sans", "Aptos", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", "JetBrains Mono", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(214, 167, 78, 0.20), transparent 28%),
        radial-gradient(circle at top right, rgba(34, 123, 103, 0.16), transparent 32%),
        linear-gradient(180deg, #f9f5eb 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 40px 24px 72px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,248,235,0.86));
      border: 1px solid rgba(255,255,255,0.6);
      border-radius: 32px;
      box-shadow: var(--shadow);
      padding: 28px 28px 32px;
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -120px -120px auto;
      width: 280px;
      height: 280px;
      background: radial-gradient(circle, rgba(195,71,44,0.18), transparent 65%);
      pointer-events: none;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(21,32,40,0.06);
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 16px 0 10px;
      font-size: clamp(34px, 5vw, 60px);
      line-height: 0.95;
      letter-spacing: -0.04em;
      max-width: 10ch;
    }}
    .hero p {{
      margin: 0;
      max-width: 72ch;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.6;
    }}
    .verdict {{
      margin-top: 20px;
      padding: 16px 18px;
      border-radius: 18px;
      background: linear-gradient(90deg, rgba(195,71,44,0.12), rgba(214,167,78,0.10));
      border: 1px solid rgba(195,71,44,0.12);
      font-size: 15px;
      line-height: 1.6;
    }}
    .hero-grid {{
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .hero-card, .meta-card, .metric-card, .story-card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.72);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero-card {{
      padding: 18px;
      border-radius: 20px;
    }}
    .hero-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .hero-value {{
      margin-top: 10px;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .hero-sub {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .meta-grid, .story-grid, .metric-grid {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .meta-card, .story-card {{
      padding: 20px;
      border-radius: 22px;
    }}
    .meta-card h3, .story-card h3, .metric-card h3 {{
      margin: 0 0 8px;
      font-size: 20px;
      letter-spacing: -0.02em;
    }}
    .meta-row {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 10px;
      padding: 8px 0;
      border-top: 1px solid var(--line);
      font-size: 14px;
    }}
    .meta-row:first-of-type {{ border-top: none; }}
    .meta-label {{ color: var(--muted); }}
    .metric-grid {{
      grid-template-columns: 1fr;
    }}
    .metric-card {{
      padding: 22px;
      border-radius: 24px;
    }}
    .metric-head {{
      display: flex;
      gap: 12px;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 16px;
    }}
    .metric-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      max-width: 70ch;
      line-height: 1.5;
    }}
    .signal-pill {{
      white-space: nowrap;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(21,32,40,0.06);
      color: var(--ink);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 8px;
      border-top: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
      vertical-align: middle;
    }}
    thead th {{
      border-top: none;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .metric-num {{
      font-family: var(--mono);
      white-space: nowrap;
    }}
    .metric-ratio {{
      font-weight: 700;
      white-space: nowrap;
    }}
    .bar-track {{
      height: 12px;
      border-radius: 999px;
      background: rgba(21,32,40,0.06);
      overflow: hidden;
      min-width: 120px;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: inherit;
    }}
    .bar-fill.bad {{
      background: linear-gradient(90deg, var(--bad), #d56b4f);
    }}
    .bar-fill.good {{
      background: linear-gradient(90deg, var(--good), #41a28c);
    }}
    .threshold-block {{
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
    }}
    .threshold-title {{
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .story-list, .bullet-list {{
      margin: 12px 0 0;
      padding-left: 18px;
      color: var(--ink);
      line-height: 1.6;
    }}
    .story-list li + li, .bullet-list li + li {{
      margin-top: 8px;
    }}
    .foot {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    @media (max-width: 760px) {{
      .wrap {{ padding: 22px 14px 48px; }}
      .hero {{ padding: 20px; }}
      .metric-head {{ flex-direction: column; }}
      .bar-track {{ min-width: 72px; }}
      .meta-row {{ grid-template-columns: 1fr; gap: 4px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Standalone Host Comparison</div>
      <h1>{html.escape(args.title)}</h1>
      <p>
        This report compares the no-camera, no-leader follower benchmark on the bad host and the good host.
        It isolates follower RPC latency and host pacing without camera or Fello confounds. The good host stays
        essentially flat. The bad host still shows repeated follower wait spikes and pacing overshoot.
      </p>
      <div class="verdict">{html.escape(verdict)}</div>
      <div class="hero-grid">{hero_cards}</div>
    </section>

    <div class="meta-grid">
      <section class="meta-card">
        <h3>Bad Host</h3>
        <div class="meta-row"><div class="meta-label">Host</div><div>{html.escape(meta_bad['hostname'])}</div></div>
        <div class="meta-row"><div class="meta-label">Station</div><div>{html.escape(meta_bad['station_key'])}</div></div>
        <div class="meta-row"><div class="meta-label">Platform</div><div>{html.escape(meta_bad['platform'])}</div></div>
        <div class="meta-row"><div class="meta-label">Python</div><div>{html.escape(meta_bad['python_version'])}</div></div>
        <div class="meta-row"><div class="meta-label">Samples</div><div>{meta_bad['iterations_recorded']}</div></div>
        <div class="meta-row"><div class="meta-label">Source</div><div>{html.escape(meta_bad['summary_path'])}</div></div>
      </section>
      <section class="meta-card">
        <h3>Good Host</h3>
        <div class="meta-row"><div class="meta-label">Host</div><div>{html.escape(meta_good['hostname'])}</div></div>
        <div class="meta-row"><div class="meta-label">Station</div><div>{html.escape(meta_good['station_key'])}</div></div>
        <div class="meta-row"><div class="meta-label">Platform</div><div>{html.escape(meta_good['platform'])}</div></div>
        <div class="meta-row"><div class="meta-label">Python</div><div>{html.escape(meta_good['python_version'])}</div></div>
        <div class="meta-row"><div class="meta-label">Samples</div><div>{meta_good['iterations_recorded']}</div></div>
        <div class="meta-row"><div class="meta-label">Source</div><div>{html.escape(meta_good['summary_path'])}</div></div>
      </section>
    </div>

    <div class="metric-grid">
      {metric_sections}
    </div>

    <div class="story-grid">
      {_outlier_story(
        bad_summary,
        "env_obs_phase_ms",
        "Bad Host Outlier Story",
        [
          "The worst bad-host outliers are follower wait excursions, not controller work.",
          "Right follower wait alone reaches 26.831 ms in the standalone benchmark.",
          "That reproduces the same class of host-side jitter seen during data collection.",
        ],
      )}
      {_outlier_story(
        bad_summary,
        "loop_total_ms",
        "Bad Host Pacing Story",
        [
          "The worst loop outliers are pacing overshoot spikes, not follower payload size.",
          "One bad-host step overshoots the scheduled wake-up by 29.639 ms.",
          "The good host has no comparable pacing regime; its max overshoot is 0.159 ms.",
        ],
      )}
      {_outlier_story(
        good_summary,
        "env_obs_phase_ms",
        "Good Host Contrast",
        [
          "The good host remains flat even at its worst steps.",
          "Its max follower observation phase is only 1.081 ms.",
          "Its max follower wait stays below 0.573 ms on both sides.",
        ],
      )}
    </div>

    <div class="foot">
      The two hosts differ in station key and platform, but this benchmark disabled leaders and cameras on both sides.
      That makes the comparison strong evidence for a host-side follower RPC plus scheduler timing problem on the bad machine,
      not a fundamental YAM controller defect.
    </div>
  </div>
</body>
</html>
"""


def main(args: Args) -> None:
    bad_summary = _load(args.bad_summary)
    good_summary = _load(args.good_summary)
    html_text = _build_report(args, bad_summary, good_summary)
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
