# ChArUco calibration board: print, mount, calibrate

Camera calibration is the one setup step that cannot be automated — it needs a
physical board, printed at a known scale, held or fixed by a person. Everything
downstream depends on it: a board printed 3% small biases every hand-eye
transform by 3%, and nothing in the pipeline will flag it, because a mis-scaled
board still detects perfectly.

Budget ~30 minutes for a first run.

---

## 1. The board this station expects

Defaults live in `enpire/env/forge/yam/calibration/config.py`:

| Property | Default | Constant |
|---|---|---|
| Columns × rows | 5 × 5 | `SQUARES_X`, `SQUARES_Y` |
| Checker square | 40 mm (`0.040`) | `SQUARE_LENGTH` |
| ArUco marker | 30 mm (`0.030`) | `MARKER_LENGTH` |
| Dictionary | `DICT_4X4_50` | `DICTIONARY` |
| Min samples | 12 | `MIN_SAMPLES` |

Printed board area is 200 × 200 mm — it fits on A4/Letter with room for a
margin.

## 2. Generate the PDF/PNG

Generate it from the repo rather than downloading one. The generator reads the
same constants the detector uses, so the board and the code cannot disagree:

```bash
uv run python -m enpire.env.forge.yam.calibration.make_board --output charuco_board.png
```

```
Wrote charuco_board.png (2360x2360 px @ 300 DPI)
Board: 5x5, square 40.0 mm, marker 30.0 mm, DICT_4X4_50
Printed size must measure 200 x 200 mm.
```

A different board — say a 7×5 with 25 mm squares — is fine, as long as you pass
the same numbers to the calibrator in step 5:

```bash
uv run python -m enpire.env.forge.yam.calibration.make_board \
  --squares-x 7 --squares-y 5 --square-length 0.025 --marker-length 0.019
```

> If you would rather use a generator website, [calib.io's pattern
> generator](https://calib.io/pages/camera-calibration-pattern-generator) can
> produce a ChArUco board. Match **all four** of columns, rows, square length,
> and marker length, and select the `4x4_50` ArUco dictionary — a board from a
> different dictionary is not detected at all, which at least fails loudly.

## 3. Print it — the step that goes wrong

1. Print at **100% / "actual size"**. Turn off "fit to page", "shrink to fit",
   and any borderless mode. This is the single most common calibration error.
2. Print on matte paper or card. Glossy stock produces specular highlights that
   destroy corner detection under room lighting.
3. **Measure a white square with calipers or a steel rule.** If it is not
   40.0 mm, do not reprint and hope — just pass the measured value in step 5.
   The calibration only needs the number to be *true*, not round.
4. Mount the sheet on something rigid and flat — foam board, acrylic, or
   aluminium. A sheet of paper that bows by a couple of millimetres injects
   error into every pose. Glue the whole face, not just the corners.
5. Keep a white margin of at least one square around the pattern. The generator
   adds this by default (`--margin-squares`); do not trim it off.

## 4. Mount it — different for each of the three steps

`station calibrate-all` runs three calibrations in sequence and prompts you
between them. The board goes in a **different place each time** — read the
prompt rather than assuming:

| Step | Camera | Where the board goes |
|---|---|---|
| 1 | `top` | **On the gripper.** The arm carries the board through poses while the fixed top camera watches. |
| 2 | `left_wrist` | **Fixed in the world**, in view of the left wrist camera. Remove it from the gripper. The arm moves the camera around the stationary board. |
| 3 | `right_wrist` | Same, for the right wrist camera. |

For step 1, attach the board to the gripper so it **cannot shift** — the whole
method assumes a rigid, unchanging board-to-gripper transform. Clamp it in the
jaws or bolt it to the wrist plate; do not tape it to one finger. If it slips
mid-run, the residuals will be poor and the run must be repeated.

For steps 2 and 3, weight or clamp the board to the table, angled so the wrist
camera sees it from a range of viewpoints. It must not move for the whole step.

Throughout: keep lighting even and diffuse, avoid a single hard lamp, and keep
the board unshadowed by the arm at the sampled poses.

## 5. Run the calibration

Clear the workspace and keep an e-stop in reach — these commands move the arm.

```bash
export ENPIRE_YAM_MODEL_ROOT=/path/to/yam-model-assets

uv run enpire station calibrate-all \
  --station my-yam-station-name \
  --output-xml /path/outside/repo/station_calibrated.xml \
  --confirm-motion
```

If your printed board differs from the defaults — including a board that
measured 39.7 mm instead of 40.0 — pass its real geometry. Any flag you omit
keeps the `config.py` default:

```bash
uv run enpire station calibrate-all \
  --station my-yam-station-name \
  --square-length 0.0397 --marker-length 0.0298 \
  --confirm-motion
```

`--squares-x`, `--squares-y`, `--square-length`, and `--marker-length` work on
both `station calibrate` (one camera) and `station calibrate-all` (all three).

The live view overlays **BOARD OK** in green when the board is detected and
**NO BOARD** in red when it is not. If it stays red, work through §7.

Validate the result without hardware, then update the station profile's
`calibration_bundle` only after the residuals and a physical sanity check pass:

```bash
uv run enpire station validate-calibration /path/outside/repo/calibration.json
```

## 6. Sanity-check the result physically

Residuals can look fine while the transform is wrong — a mis-scaled board
produces a self-consistent but incorrect fit. Before trusting the calibration,
command the arm to a known point on the table and confirm the tool lands where
the camera says it does. A systematic offset that grows with distance from the
base is the signature of a board-scale error.

## 7. When the board is not detected

| Symptom | Likely cause |
|---|---|
| `NO BOARD` everywhere | Wrong dictionary — a `DICT_5X5`/`DICT_6X6` board never matches `DICT_4X4_50`. Regenerate with the generator. |
| Detects, but poses are biased | Printed at the wrong scale. Measure a square and pass `--square-length`. |
| Detects only head-on | Glossy paper, or a single hard light source. Use matte stock and diffuse lighting. |
| Corner count fluctuates | Board is bowed, or too far from the camera. Flatten it; move it closer. |
| Detects, hand-eye residuals poor | Board moved relative to the gripper mid-run (step 1), or moved on the table (steps 2–3). Re-mount rigidly and repeat. |
| Too few samples accepted | Fewer than `MIN_SAMPLES` (12) valid poses. Give the board more viewpoint variety. |

---

## Related

- `enpire/env/docs/REAL_WORLD_WORKFLOWS.md` — full station bring-up sequence
- `enpire/env/docs/MULTI_CAMERA_CONFIG.md` — camera roles and multi-camera setup
- `README.md` — binding cameras to roles (`/dev/video_*`, aliases, udev)
