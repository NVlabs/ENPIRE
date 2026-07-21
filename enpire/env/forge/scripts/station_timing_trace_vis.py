from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


PREFERRED_METRICS = [
    "env_obs_phase_ms",
    "left_follower_wait_ms",
    "right_follower_wait_ms",
    "obs_component_span_ms",
    "pacing_overshoot_ms",
    "loop_total_ms",
    "pacing_ms",
    "policy_phase_ms",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _detect_metrics(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows[:50]:
        for key, value in row.items():
            if key == "step":
                continue
            if not str(key).endswith("_ms"):
                continue
            if isinstance(value, (int, float)):
                keys.add(str(key))
    ordered = [k for k in PREFERRED_METRICS if k in keys]
    ordered.extend(sorted(k for k in keys if k not in ordered))
    return ordered


def _series(rows: list[dict[str, Any]], metric: str) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row in rows:
        value = row.get(metric)
        step = row.get("step")
        if not isinstance(step, (int, float)) or not isinstance(value, (int, float)):
            continue
        result.append({
            "step": float(step),
            "value": float(value),
        })
    return result


def _top_spikes(rows: list[dict[str, Any]], metric: str, topn: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(
        [row for row in rows if isinstance(row.get(metric), (int, float))],
        key=lambda row: float(row.get(metric, 0.0)),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in ranked[:topn]:
        out.append({
            "step": int(row.get("step", 0)),
            "wall_time": row.get("wall_time_iso", ""),
            "value": round(float(row.get(metric, 0.0)), 3),
            "loop_total_ms": round(float(row.get("loop_total_ms", 0.0)), 3),
            "env_obs_phase_ms": round(float(row.get("env_obs_phase_ms", 0.0)), 3),
            "pacing_overshoot_ms": round(float(row.get("pacing_overshoot_ms", 0.0)), 3),
            "left_follower_wait_ms": round(float(row.get("left_follower_wait_ms", 0.0)), 3),
            "right_follower_wait_ms": round(float(row.get("right_follower_wait_ms", 0.0)), 3),
            "obs_component_span_ms": round(float(row.get("obs_component_span_ms", 0.0)), 3),
        })
    return out


def build_report(
    bad_rows: list[dict[str, Any]],
    good_rows: list[dict[str, Any]],
    bad_path: Path,
    good_path: Path,
    *,
    bad_label: str,
    good_label: str,
) -> str:
    metrics = _detect_metrics(bad_rows + good_rows)
    if not metrics:
        raise ValueError("No *_ms metrics found in the supplied raw JSONL files")

    payload = {
        "bad_label": bad_label,
        "good_label": good_label,
        "bad_path": str(bad_path),
        "good_path": str(good_path),
        "metrics": metrics,
        "bad_rows": bad_rows,
        "good_rows": good_rows,
    }
    payload_json = json.dumps(payload, ensure_ascii=True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Station Timing Iteration Trace</title>
  <style>
    :root {{
      --bg: #09111f;
      --panel: rgba(15, 23, 42, 0.88);
      --line: rgba(148, 163, 184, 0.15);
      --text: #ebf2ff;
      --muted: #93a6c4;
      --bad: #fb7185;
      --good: #2dd4bf;
      --accent: #60a5fa;
      --shadow: 0 20px 48px rgba(2, 6, 23, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 18%, rgba(96,165,250,0.14), transparent 28%),
        radial-gradient(circle at 84% 16%, rgba(45,212,191,0.10), transparent 22%),
        radial-gradient(circle at 80% 80%, rgba(251,113,133,0.10), transparent 24%),
        linear-gradient(180deg, #09111f 0%, #070c17 100%);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 28px 22px 42px; }}
    .hero, .card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero {{
      padding: 24px 26px 20px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      letter-spacing: -0.04em;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.02;
    }}
    .subtitle {{
      color: var(--muted);
      line-height: 1.6;
      max-width: 1100px;
      font-size: 15px;
    }}
    .pill-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .pill {{
      padding: 6px 11px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      color: #dbeafe;
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.45fr 0.95fr;
      gap: 18px;
    }}
    .card {{ padding: 18px; }}
    h2 {{
      margin: 0 0 14px;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      letter-spacing: -0.03em;
      font-size: 19px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .toolbar-label {{
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }}
    .segmented {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .segmented button {{
      appearance: none;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }}
    .segmented button.active {{
      background: rgba(96,165,250,0.16);
      border-color: rgba(96,165,250,0.42);
      color: #dbeafe;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .swatch.bad {{ background: var(--bad); }}
    .swatch.good {{ background: var(--good); }}
    .chart-caption {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      margin-bottom: 10px;
    }}
    .chart-svg {{
      width: 100%;
      height: 430px;
      display: block;
      overflow: visible;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }}
    .tick-label {{
      fill: rgba(226,232,240,0.70);
      font-size: 11px;
    }}
    .grid-line {{
      stroke: rgba(148,163,184,0.12);
      stroke-width: 1;
    }}
    .spike-point {{
      cursor: pointer;
    }}
    .detail-grid {{
      display: grid;
      gap: 10px;
    }}
    .detail-hero {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .detail-stat {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(148,163,184,0.08);
    }}
    .detail-stat .k {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .detail-stat .v {{
      font-size: 22px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .spike-list {{
      display: grid;
      gap: 10px;
      margin-top: 8px;
    }}
    .spike-item {{
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(148,163,184,0.08);
      padding: 12px;
      display: grid;
      gap: 7px;
    }}
    .spike-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      font-variant-numeric: tabular-nums;
    }}
    .spike-host.bad {{ color: #fecdd3; }}
    .spike-host.good {{ color: #99f6e4; }}
    .muted {{ color: var(--muted); }}
    .tooltip {{
      position: fixed;
      z-index: 50;
      pointer-events: none;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(7,12,24,0.95);
      border: 1px solid rgba(148,163,184,0.18);
      color: var(--text);
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 16px 32px rgba(2,6,23,0.4);
      max-width: 280px;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 100ms ease, transform 100ms ease;
    }}
    .tooltip.visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .detail-hero {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Raw iteration trace</div>
      <h1>Iteration vs timing spike view</h1>
      <div class="subtitle">
        This report overlays the same raw benchmark metric from both machines so you can see exactly where the bad host spikes.
        X is iteration. Y is the selected timing metric in milliseconds. Click a metric button to switch the view.
      </div>
      <div class="pill-row">
        <span class="pill">{html.escape(str(bad_path))}</span>
        <span class="pill">{html.escape(str(good_path))}</span>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Iteration trace</h2>
        <div class="toolbar">
          <div class="toolbar-label">Metric</div>
          <div id="metric-toggle" class="segmented"></div>
        </div>
        <div class="legend">
          <div class="legend-item"><span class="swatch bad"></span>{html.escape(bad_label)}</div>
          <div class="legend-item"><span class="swatch good"></span>{html.escape(good_label)}</div>
        </div>
        <div class="chart-caption">
          The chart highlights the top spikes for the selected metric. Hover a spike marker for its exact iteration and timing.
        </div>
        <svg id="trace-chart" class="chart-svg" viewBox="0 0 960 430" preserveAspectRatio="none"></svg>
      </article>

      <article class="card">
        <h2>Spike summary</h2>
        <div class="chart-caption">
          Worst spikes for the selected metric on each machine, with supporting timing buckets from the same raw rows.
        </div>
        <div id="detail" class="detail-grid"></div>
      </article>
    </section>
  </div>
  <div id="tooltip" class="tooltip"></div>

  <script id="trace-data" type="application/json">{payload_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('trace-data').textContent);
    const TOOLTIP = document.getElementById('tooltip');
    const STATE = {{
      metric: DATA.metrics[0],
    }};

    function esc(text) {{
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }}

    function fmtMs(v) {{
      return Number.isFinite(v) ? `${{v.toFixed(3)}} ms` : 'n/a';
    }}

    function fmtNum(v) {{
      return Number.isFinite(v) ? `${{Math.round(v)}}` : 'n/a';
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

    function rowsFor(host) {{
      return host === 'bad' ? DATA.bad_rows : DATA.good_rows;
    }}

    function series(host, metric) {{
      return rowsFor(host)
        .filter(row => Number.isFinite(Number(row.step)) && Number.isFinite(Number(row[metric])))
        .map(row => ({{
          step: Number(row.step),
          value: Number(row[metric]),
          row,
        }}));
    }}

    function buildPath(points, x, y) {{
      if (!points.length) return '';
      return points.map((point, idx) => `${{idx === 0 ? 'M' : 'L'}}${{x(point.step).toFixed(2)}},${{y(point.value).toFixed(2)}}`).join(' ');
    }}

    function renderMetricToggle() {{
      const el = document.getElementById('metric-toggle');
      el.innerHTML = DATA.metrics.map(metric => `
        <button class="${{metric === STATE.metric ? 'active' : ''}}" data-metric="${{metric}}">
          ${{metric.replaceAll('_ms', '').replaceAll('_', ' ')}}
        </button>
      `).join('');
      el.querySelectorAll('button').forEach(button => {{
        button.addEventListener('click', () => {{
          STATE.metric = button.dataset.metric;
          render();
        }});
      }});
    }}

    function renderTrace() {{
      const bad = series('bad', STATE.metric);
      const good = series('good', STATE.metric);
      const svg = document.getElementById('trace-chart');
      const width = 960;
      const height = 430;
      const pad = {{ left: 72, right: 20, top: 20, bottom: 48 }};
      const maxStep = Math.max(1, ...bad.map(p => p.step), ...good.map(p => p.step));
      const maxValue = Math.max(1e-6, ...bad.map(p => p.value), ...good.map(p => p.value)) * 1.08;
      const x = value => pad.left + (value / maxStep) * (width - pad.left - pad.right);
      const y = value => height - pad.bottom - (value / maxValue) * (height - pad.top - pad.bottom);
      const ticks = Array.from({{ length: 5 }}, (_, i) => (maxValue * i) / 4);
      const badTop = bad.slice().sort((a, b) => b.value - a.value).slice(0, 10);
      const goodTop = good.slice().sort((a, b) => b.value - a.value).slice(0, 10);

      svg.innerHTML = `
        ${{ticks.map(t => `
          <line class="grid-line" x1="${{pad.left}}" y1="${{y(t)}}" x2="${{width - pad.right}}" y2="${{y(t)}}"></line>
          <text class="tick-label" x="58" y="${{y(t) + 4}}" text-anchor="end">${{t.toFixed(1)}}</text>
        `).join('')}}
        <text class="axis-label" x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle">iteration</text>
        <text class="axis-label" x="18" y="${{height / 2}}" text-anchor="middle" transform="rotate(-90 18 ${{height / 2}})">${{STATE.metric}}</text>
        <path d="${{buildPath(good, x, y)}}" fill="none" stroke="rgba(45,212,191,0.92)" stroke-width="2.1"></path>
        <path d="${{buildPath(bad, x, y)}}" fill="none" stroke="rgba(251,113,133,0.95)" stroke-width="2.1"></path>
        ${{badTop.map((point, idx) => `
          <circle class="spike-point" data-host="bad" data-index="${{idx}}" cx="${{x(point.step)}}" cy="${{y(point.value)}}" r="4.8" fill="rgba(251,113,133,0.98)"></circle>
        `).join('')}}
        ${{goodTop.map((point, idx) => `
          <circle class="spike-point" data-host="good" data-index="${{idx}}" cx="${{x(point.step)}}" cy="${{y(point.value)}}" r="4.8" fill="rgba(45,212,191,0.98)"></circle>
        `).join('')}}
      `;

      const lookup = {{ bad: badTop, good: goodTop }};
      svg.querySelectorAll('.spike-point').forEach(node => {{
        const host = node.dataset.host;
        const point = lookup[host][Number(node.dataset.index)];
        node.addEventListener('mouseenter', evt => showTooltip(evt, `
          <strong>${{host === 'bad' ? esc(DATA.bad_label) : esc(DATA.good_label)}}</strong><br>
          iteration: ${{fmtNum(point.step)}}<br>
          value: ${{fmtMs(point.value)}}<br>
          loop: ${{fmtMs(Number(point.row.loop_total_ms || 0))}}<br>
          env: ${{fmtMs(Number(point.row.env_obs_phase_ms || 0))}}<br>
          pace overshoot: ${{fmtMs(Number(point.row.pacing_overshoot_ms || 0))}}
        `));
        node.addEventListener('mousemove', moveTooltip);
        node.addEventListener('mouseleave', hideTooltip);
      }});

      renderDetail(badTop, goodTop);
    }}

    function spikeItem(hostClass, hostLabel, point) {{
      return `
        <div class="spike-item">
          <div class="spike-head">
            <div class="spike-host ${{hostClass}}">${{hostLabel}} step ${{fmtNum(point.step)}}</div>
            <div>${{fmtMs(point.value)}}</div>
          </div>
          <div class="muted">loop ${{fmtMs(Number(point.row.loop_total_ms || 0))}} | env ${{fmtMs(Number(point.row.env_obs_phase_ms || 0))}} | pace ${{fmtMs(Number(point.row.pacing_overshoot_ms || 0))}}</div>
          <div class="muted">left follower ${{fmtMs(Number(point.row.left_follower_wait_ms || 0))}} | right follower ${{fmtMs(Number(point.row.right_follower_wait_ms || 0))}} | span ${{fmtMs(Number(point.row.obs_component_span_ms || 0))}}</div>
        </div>
      `;
    }}

    function renderDetail(badTop, goodTop) {{
      const detail = document.getElementById('detail');
      detail.innerHTML = `
        <div class="detail-hero">
          <div class="detail-stat">
            <div class="k">Selected metric</div>
            <div class="v" style="font-size:16px;">${{esc(STATE.metric)}}</div>
          </div>
          <div class="detail-stat">
            <div class="k">Bad / good max</div>
            <div class="v">${{fmtMs(badTop[0] ? badTop[0].value : NaN)}} / ${{fmtMs(goodTop[0] ? goodTop[0].value : NaN)}}</div>
          </div>
        </div>
        <div class="spike-list">
          ${{badTop.slice(0, 5).map(point => spikeItem('bad', esc(DATA.bad_label), point)).join('')}}
          ${{goodTop.slice(0, 5).map(point => spikeItem('good', esc(DATA.good_label), point)).join('')}}
        </div>
      `;
    }}

    function render() {{
      renderMetricToggle();
      renderTrace();
    }}

    render();
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an iteration-vs-time comparison from raw benchmark JSONL")
    parser.add_argument("bad_raw", type=Path, help="Path to bad-host raw JSONL")
    parser.add_argument("good_raw", type=Path, help="Path to good-host raw JSONL")
    parser.add_argument("-o", "--output", type=Path, default=Path("station_timing_trace.html"))
    parser.add_argument("--bad-label", default="bad host")
    parser.add_argument("--good-label", default="good host")
    args = parser.parse_args()

    bad_rows = _load_jsonl(args.bad_raw)
    good_rows = _load_jsonl(args.good_raw)
    html_text = build_report(
        bad_rows,
        good_rows,
        args.bad_raw,
        args.good_raw,
        bad_label=args.bad_label,
        good_label=args.good_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote iteration trace report to {args.output}")


if __name__ == "__main__":
    main()
