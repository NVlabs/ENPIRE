# RoboCasa Oracle API

Structured oracle targets for scripted / oracle-mode runs in RoboCasa.

## Enabling

Set `oracle_api: true` in your experiment YAML. This adds `get_oracle_targets`
to the tool namespace, available in both agent-generated code and oracle scripts.

```yaml
oracle_api: true
```

## `get_oracle_targets() → dict`

Returns a dict describing the current task's actionable target geometry, derived
directly from the MuJoCo sim. All positions are in **world frame**.

### Common fields (all tasks)

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | str | Task class name, e.g. `"OpenCabinet"` |
| `supported` | bool | False if task type is not recognised |
| `reason` | str | Error message when `supported=False` |
| `fixture` | dict | Fixture body pose + name |
| `target` | dict | Primary actionable target (see per-task below) |
| `controls` | dict[str, dict] | All actionable controls (e.g. both cabinet handles) |
| `active_control` | str | Which control in `controls` has most remaining progress |
| `fixture_state` | dict | Live joint / semantic state of the fixture |
| `task_info` | dict | Snapshot of `get_task_info()` at call time |

### Target `kind` values and their fields

#### `"press"` — button / switch
```
pos, surface_normal, recommended_standoff, recommended_press_distance, recommended_retreat_distance
```
Tasks: `TurnOnMicrowave`, `TurnOffMicrowave`, `TurnOnElectricKettle`

#### `"turn"` — rotary knob / faucet handle
```
pos, surface_normal, axis_world, anchor_world, normalized_qpos, range,
recommended_standoff, recommended_turn_amount, recommended_retreat_distance
```
Tasks: `TurnOnSinkFaucet`, `TurnOffSinkFaucet`, `TurnOnStove`, `TurnOffStove`, `LowerHeat`

#### `"slider_handle"` — translating drawer / rack
```
pos, surface_normal, axis_world, anchor_world, normalized_qpos, desired_fraction,
recommended_standoff, recommended_contact_offset, recommended_travel_distance, recommended_retreat_distance
```
Tasks: `OpenDrawer`, `SlideDishwasherRack`

#### `"hinge_handle"` — swinging cabinet / fridge door
```
pos, surface_normal, axis_world, anchor_world, normalized_qpos, desired_fraction,
recommended_standoff, recommended_contact_offset, recommended_travel_distance, recommended_retreat_distance
```
Tasks: `OpenCabinet`, `CloseFridge`, `CloseToasterOvenDoor`, `OpenStandMixerHead`

#### `"pick_place"` — pick an object and place it at target
```
pos, quat_xyzw, recommended_pick_standoff, recommended_place_standoff
```
Tasks: `CloseBlenderLid`, `CoffeeSetupMug`

#### `"base_pose"` — mobile base navigation target
```
pos, yaw
```
Tasks: `NavigateKitchen`

## Oracle scripts

| Script | Task |
|--------|------|
| `cap/saved_scripts/open_cabinet_human.py` | `OpenCabinet` |
| `cap/saved_scripts/pnp_microwave_human.py` | `TurnOnMicrowave` (via `get_oracle_targets`) |

## Implementation

`get_oracle_targets()` is implemented in `cap/env/robocasa/env.py` as
`RoboCasaEnv.get_oracle_targets()`, using MuJoCo model/data queries to extract
geom poses, joint state, and fixture-specific metadata.
