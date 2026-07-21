# RoboCasa COMPOSITE_SEEN Vision Scripts

Vision-based CAP scripts for all 16 RoboCasa COMPOSITE_SEEN benchmark tasks. Each script follows the canonical `_vision.py` template established by `pnp_counter_to_cabinet_vision.py`: install the `skill_library.namespace` shim, import helpers from `cap/saved_scripts/robocasa_skill_library/`, run a multi-stage detect → grasp → place → verify flow, and `raise RuntimeError` if `get_task_info()['success']` is False.

These scripts use only camera detection (SAM3 + depth) and runtime APIs (`get_robot_state`, `get_task_info`, `get_task_description`). No simulator ground truth is read — they are written to be transferable to physical perception when ready.

## Tasks and entry points

Launch any task with:

```bash
uv run python run_agent.py experiment=<name>
```

| Task class | Experiment | Vision script | New helpers |
|---|---|---|---|
| PrepareCoffee | `prepare_coffee` | `prepare_coffee_vision.py` | `coffee_machine_control` |
| KettleBoiling | `kettle_boiling` | `kettle_boiling_vision.py` | `pan_burner_alignment` |
| PreSoakPan | `pre_soak_pan` | `pre_soak_pan_vision.py` | — (reuses sink_faucet_motion + pan_burner_alignment) |
| WashLettuce | `wash_lettuce` | `wash_lettuce_vision.py` | `under_water_holding` |
| RinseSinkBasin | `rinse_sink_basin` | `rinse_sink_basin_vision.py` | `sink_spout_rotation` |
| DeliverStraw | `deliver_straw` | `deliver_straw_vision.py` | `thin_object_insertion` |
| StackBowlsCabinet | `stack_bowls_cabinet` | `stack_bowls_cabinet_vision.py` | `bowl_stacking` |
| SetUpCuttingStation | `set_up_cutting_station` | `set_up_cutting_station_vision.py` | — (reuses bowl_stacking placement) |
| StoreLeftoversInBowl | `store_leftovers_in_bowl` | `store_leftovers_in_bowl_vision.py` | — |
| SearingMeat | `searing_meat` | `searing_meat_vision.py` | `pan_burner_alignment` (full) |
| StirVegetables | `stir_vegetables` | `stir_vegetables_vision.py` | `stirring_motion` |
| ScrubCuttingBoard | `scrub_cutting_board` | `scrub_cutting_board_vision.py` | `scrubbing_motion`, `release_with_clearance_v1` (in `arm_motion`) |
| GetToastedBread | `get_toasted_bread` | `get_toasted_bread_vision.py` | `toaster_lever_press_v1`, `toaster_wait_until_popped_v1` (in `toaster_extract`) |
| SteamInMicrowave | `steam_in_microwave` | `steam_in_microwave_vision.py` | `microwave_control` |
| LoadDishwasher | `load_dishwasher` | `load_dishwasher_vision.py` | `dishwasher_control` |
| PackIdenticalLunches | `pack_identical_lunches` | `pack_identical_lunches_vision.py` | — (uses bowl_stacking + dedup logic in script) |

## New skill-library helpers

All under `cap/saved_scripts/robocasa_skill_library/`. Each module begins with `from skill_library.namespace import *` and follows the existing `_v1` versioning convention.

- `coffee_machine_control.py` — detect dispenser cradle, place mug under it, detect + press start button.
- `pan_burner_alignment.py` — detect stove top + burner sites, choose target burner (rightmost / closest to arm / highest score), top-down place a held pan/kettle on chosen burner with XY fallback offsets.
- `under_water_holding.py` — hold a grasped object below the faucet stream and poll `get_task_info()` until success or timeout.
- `sink_spout_rotation.py` — detect spout, grasp, sweep through left/center/right positions polling success.
- `thin_object_insertion.py` — detect cup mouth (centroid + heuristic Z bump), insert held thin object via hover + descend with XY-candidate offsets.
- `bowl_stacking.py` — detect 2+ bowls, plan a stack target above the lower bowl, generic `place_bowl_at_v1` reused across many scripts as a "place held object at xyz" primitive.
- `scrubbing_motion.py` — detect cutting board, plan zigzag scrub waypoints (≥6 contacts spanning ≥0.12m), execute with `nudge_brutal` per segment, polling success.
- `stirring_motion.py` — detect pot interior, plan circular stir waypoints (radius 0.05m × 4 revolutions × 8 points), execute with success-tick counter to require ≥5 sustained ticks.
- `microwave_control.py` — detect microwave + door + start button, push door closed with body-relative direction, press start with XY fallbacks.
- `dishwasher_control.py` — detect dishwasher + top rack + door, push rack in toward dishwasher body, push door upward+inward to close.

Plus three additions to existing modules:

- `arm_motion.py` → `release_with_clearance_v1(side, clearance_m)` — open gripper + brutal-nudge upward; satisfies `gripper_obj_far` checks (default threshold 0.15m).
- `toaster_extract.py` → `toaster_lever_press_v1`, `toaster_wait_until_popped_v1` — start toasting and poll `get_task_info()` until done.
- `stove_knob_detection.py` → `detect_turn_off_stove_knobs_v1` now accepts an explicit `target_knob` kwarg (skipping task-description parsing) so KettleBoiling/SearingMeat/StirVegetables can target `front_right` directly without the task naming a knob.

## Architecture conventions preserved

Every new vision script:

1. Installs the synthetic `skill_library.namespace` module from its globals via `_install_skill_namespace()` so helpers can `from skill_library.namespace import *`.
2. Imports only from `cap.saved_scripts.robocasa_skill_library.*` (the canonical RoboCasa library), not from the legacy `cap.saved_scripts.skill_library`.
3. Uses `nudge_brutal` (not `nudge`) when the planner sees the EE inside a fixture (toaster slot, microwave interior, scrub-on-board, stir-in-pot, dishwasher rack push, microwave door close).
4. Calls `disable_collision_avoid()` before final `go_home` when the arm is near a fixture.
5. Ends with `raise RuntimeError(...)` on failure so the agent runtime triggers reflection.

## Known risks and mitigations

- **Microwave start button localization** — start button is a single geom on the panel; SAM3 may detect "panel" not "button". The helper has a heuristic fallback (offset right of microwave centroid) when SAM3 returns no candidates.
- **Scrub contact counter** — RoboCasa requires ≥5 distinct contact positions ≥0.02m apart spanning ≥0.1m. The default plan uses 8 contacts spanning 0.14m to give margin.
- **Stir 5-conjunct success** — must keep spatula grasped (no `open_gripper` after stirring), spatula tip inside pot rim, vegs in pot, ≥0.05cm/tick movement, pot on lit burner. The script intentionally does not release the spatula at the end.
- **Pack lunches duplicate avoidance** — script tracks already-grasped XY positions per item kind and excludes future detections within 0.06m to avoid grasping the same object twice.
- **Searing meat pan orientation** — pan spawns sideways in the cabinet; current script does not actively reorient to flat. May need an oriented-place follow-up if pan tilts on burner.

## Related docs

- `SKILL_LIBRARY_ROBOCASA.md` — broader skill library overview (older helpers).
- `ROBOCASA_INTEGRATION.md` — RoboCasaEnv architecture, OSC controller, camera system.
- `ROBOCASA_RANDOMNESS.md` — layout/style/seed determinism (relevant for repeatable evaluation across the 16 tasks).
- `AGENT_PIPELINE_DESIGN.md` — `run_agent.py` pipeline that loads each script.
