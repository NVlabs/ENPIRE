# Real prompted pickup

This example composes the existing text-prompted segmentation, grasp ranking,
cuRobo planning, and YAM control tools. The single saved script contains no
object-specific logic, station pose, camera serial, credential, or model path.

First start or configure the perception/planning services, validate the
station, clear the workspace, and keep an emergency stop within reach. Then:

```bash
uv run enpire cap run pickup --prompt "blue cube" \
  --station my-yam --confirm-motion
```

The example defaults to the AnyGrasp 6-DoF backend. If the workstation does
not have a valid AnyGrasp hardware license, select the calibrated top-down
grasp sampler instead:

```bash
ENPIRE_PICK_GRASP_MODE=2d \
uv run enpire cap run pickup --prompt "blue cube" \
  --station my-yam --confirm-motion
```

Inspect the exact command without moving hardware:

```bash
uv run enpire cap run pickup --prompt "blue cube" \
  --station my-yam --dry-run
```
