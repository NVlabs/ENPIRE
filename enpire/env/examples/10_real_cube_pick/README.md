# Real cube pickup

This example composes the existing segmentation, AnyGrasp, grasp ranking,
cuRobo planning, and YAM control tools. It intentionally contains no station
pose, camera serial, credential, or model path.

First start or configure the perception/planning services, validate the
station, clear the workspace, and keep an emergency stop within reach. Then:

```bash
uv run enpire cap run cube-pick --station my-yam --confirm-motion
```

The example defaults to the AnyGrasp 6-DoF backend. If the workstation does
not have a valid AnyGrasp hardware license, select the existing calibrated
top-down grasp sampler instead:

```bash
ENPIRE_CUBE_GRASP_MODE=2d \
uv run enpire cap run cube-pick --station my-yam --confirm-motion
```

Inspect the exact command without moving hardware:

```bash
uv run enpire cap run cube-pick --station my-yam --dry-run
```
