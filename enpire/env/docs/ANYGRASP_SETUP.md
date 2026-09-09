# AnyGrasp setup

AnyGrasp is the 6-DoF grasp-proposal backend used by `enpire cap run pickup`. It
is the only dependency ENPIRE cannot install for you, and the only one that
requires a per-machine licence.

**You may not need it.** For flat or short objects on a table, the built-in 2D
top-down sampler usually works *better* — jump to [§6](#6-the-2d-alternative).

---

## 1. Why ENPIRE does not ship it

AnyGrasp is proprietary software from GraspNet. Its SDK, model checkpoint, and
licence archive are **not** redistributable, so this repository contains none of
them — only the client code that calls into them.

The licence is also **machine-locked**: GraspNet issues it against a *feature ID*
derived from the specific machine that will run it. A licence is therefore not
portable. Each workstation, each GPU server, and each new machine after a
hardware change needs its own. Plan for this in advance — approval is not
instant (see §2).

## 2. Applying for a licence

Applications go to the upstream project, not to NVIDIA. From
[graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk):

1. Clone the SDK and generate your machine's **feature ID**, following
   `license_registration/README.md` in that repo.
2. Submit the application form: <https://forms.gle/XVV3Eip8njTYJEBo6>, supplying
   that feature ID.
3. Upstream states they *"usually reply in 5 workdays"* — and to check your spam
   folder if nothing arrives.

You receive a licence archive (a `.zip`). Keep it outside this repository;
`SECURITY.md` forbids committing licensed assets, and `.gitignore`/LFS rules
here do not cover it for you.

> Apply for every machine you intend to run on, at the same time. Discovering
> on demo day that the second workstation has no licence costs another week.

## 3. Building the SDK

ENPIRE does not build AnyGrasp. Follow the upstream instructions; in outline:

```bash
# 1. PyTorch matching your CUDA version (must match the toolkit you build with)
# 2. MinkowskiEngine v0.5.4 — use the branch matching your CUDA
# 3. SDK Python requirements
pip install -r requirements.txt
# 4. pointnet2 CUDA extension
cd pointnet2 && python setup.py install
# 5. graspnetAPI
```

Two compiled artifacts are the actual output, and ENPIRE looks for both:

- `pointnet2/build/lib.linux-x86_64-cpython-311/pointnet2/_ext.cpython-311-x86_64-linux-gnu.so`
- `dependencies/MinkowskiEngine/build/lib.linux-x86_64-cpython-311/MinkowskiEngineBackend/_C.cpython-311-x86_64-linux-gnu.so`

Note `cpython-311`: build against **Python 3.11**, the version ENPIRE uses.
Extensions built for another Python will not load, and the resulting error names
the missing path rather than the version mismatch.

This step is per-machine too — the extensions are compiled against the local
CUDA and Python.

## 4. Pointing ENPIRE at it

Set these in the station environment file, not inline per command:

| Variable | Default | Meaning |
|---|---|---|
| `ANYGRASP_SDK_ROOT` | `third_party/anygrasp_sdk` | Your SDK checkout with the built `.so` files |
| `ANYGRASP_CHECKPOINT` | `<repo>/checkpoint_detection.tar` | Detection checkpoint from the SDK |
| `ANYGRASP_LICENSE_ZIP` | *(empty)* | Your licence archive — **must be set** |
| `ANYGRASP_RUNTIME_DIR` | `/tmp/anygrasp_sdk_runtime` | Scratch dir for symlinked binaries and the extracted licence |
| `ANYGRASP_SERVER_HOST` | `localhost` | Where the client looks for the server |
| `ANYGRASP_SERVER_PORT` | `8122` | " |
| `ANYGRASP_MIN_PLANNER_Z_M` | `0.80` | Floor on proposed grasp height, in base frame |
| `ANYGRASP_MINKOWSKI_BACKEND` | unset | Override path to the compiled MinkowskiEngine extension |
| `ANYGRASP_BINARY_SEARCH_ROOTS` | unset | Extra roots to search for the native binaries |

```bash
export ANYGRASP_SDK_ROOT=/opt/anygrasp_sdk
export ANYGRASP_CHECKPOINT=/opt/anygrasp/checkpoint_detection.tar
export ANYGRASP_LICENSE_ZIP=/opt/anygrasp/license_<yourname>.zip
```

At startup ENPIRE symlinks `gsnet.so`, `lib_cxx.so`, and `tracker.so` into
`ANYGRASP_RUNTIME_DIR` and extracts the licence archive there
(`enpire/env/forge/cap/utils/anygrasp_runtime.py`).

## 5. Starting and verifying the service

```bash
uv run enpire services start --profile cap-real \
  --station my-yam-station-name --confirm-motion
```

`cap-real` is `sam3 + anygrasp + curobo + nvidia`. The AnyGrasp pane runs
`python -m enpire.env.forge.tools.vision.serve_anygrasp --host 127.0.0.1 --port 8122`.

> `services start` does **not** preflight these variables. If the licence or
> checkpoint is missing, the tmux pane dies and the CLI still reports success —
> so check the pane rather than trusting the command:
>
> ```bash
> ss -ltn | grep 8122
> tmux capture-pane -p -t enpire:anygrasp | tail -20
> ```

### Failure messages

| Message | Meaning |
|---|---|
| `AnyGrasp license zip not found: .` | `ANYGRASP_LICENSE_ZIP` is unset — this is the default-empty case. |
| `AnyGrasp license zip not found: <path>` | Set, but wrong path. |
| `Missing AnyGrasp binary for this Python: <path>` | SDK not built, or built for a different Python. |
| `Could not find prebuilt <kind> package root under <dir>` | `pointnet2` / MinkowskiEngine build missing or in an unexpected layout. |
| `Could not prepare MinkowskiEngine runtime` | No usable `build/lib.*` root; set `ANYGRASP_MINKOWSKI_BACKEND`. |
| An unlabelled crash inside `AnyGrasp(cfg)` | **Licence rejected** — expired, or issued for a different machine. This error comes from closed-source `gsnet.so`; ENPIRE cannot annotate it. Re-check that the feature ID matches this host. |
| `Cannot reach AnyGrasp server at <url>` (client side) | Server not up; see above. |
| Client timeout | Cold start on first request; raise `ANYGRASP_PLAN_TIMEOUT_S`. |

## 6. The 2D alternative

If you have no licence — or are waiting on one — the calibrated top-down
sampler needs no licence, no SDK, and no CUDA build. It depends only on
segmentation plus a known table plane:

```bash
uv run enpire services start --services sam3,curobo,yam \
  --station my-yam-station-name --confirm-motion

ENPIRE_PICK_GRASP_MODE=2d ENPIRE_PICK_CAMERA=top \
uv run enpire cap run pickup --prompt "blue cube" \
  --station my-yam-station-name --confirm-motion
```

For flat and short objects this is not merely a fallback — a top-down approach
is the correct grasp for them, and unlike point-cloud methods it does not depend
on depth, which matters on a short-baseline camera such as the D405. See the
"2D top-down grasps" section of the top-level `README.md` for the grasp-height
variables (`TABLE_SURFACE_Z_M`, `ENPIRE_2D_GRASP_Z_OFFSET_M`).

---

## Related

- [`INSTALL.md`](INSTALL.md) — install matrix
- [`CUROBO_SETUP.md`](CUROBO_SETUP.md) — the other externally-supplied dependency
- [`grasp_orientation.md`](grasp_orientation.md) — grasp frame conventions
- `THIRD_PARTY_LICENSES.md` — licensing status of bundled and external components
