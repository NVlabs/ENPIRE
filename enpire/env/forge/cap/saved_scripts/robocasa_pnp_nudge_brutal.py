"""Pick-and-place with nudge_brutal for sink escape.

Identical to the iter_002 assembly that scored 4/8, with one change:
the "test lift" nudge after grasp (step 5b) is replaced with nudge_brutal
so the arm can escape the sink wall contact even when cuRobo refuses to plan.
"""

import numpy as np
from skill_library.hover_above import hover_above_v1
from skill_library.vertical_grasp import vertical_grasp_v1
from skill_library.nudge_down_and_regrasp import nudge_down_and_regrasp_v1
from skill_library.lift import lift_v2
from skill_library.vertical_place import vertical_place_v1

SIDE = "right"

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
obj_name = info.get("obj_name", "unknown")

# Determine place target: container_pos > distr_counter_pos > distr_cab_pos
if "container_pos" in info and info["container_pos"] is not None:
    place_pos = np.array(info["container_pos"])
    place_name = info.get("container_name", "container")
elif "distr_counter_pos" in info and info["distr_counter_pos"] is not None:
    place_pos = np.array(info["distr_counter_pos"])
    place_name = info.get("distr_counter_name", "counter")
else:
    place_pos = np.array(info["distr_cab_pos"])
    place_name = "cabinet"

print(f"Task: pick '{obj_name}' at {obj_pos.tolist()} -> place on '{place_name}' at {place_pos.tolist()}")

small_objects = ("egg", "garlic", "mushroom", "lemon_wedge", "lemon", "lime", "cherry_tomato")
if obj_name in small_objects:
    grasp_z_offset = -0.02
    nudge_delta = -0.025
    print(f"Small object '{obj_name}': z_offset={grasp_z_offset}, nudge_delta={nudge_delta}")
else:
    grasp_z_offset = -0.005
    nudge_delta = -0.02
    print(f"Object '{obj_name}': z_offset={grasp_z_offset}, nudge_delta={nudge_delta}")

# Step 1: Open gripper
open_gripper(SIDE)
print("Gripper opened")

# Step 2: Hover above object
s_hover, log = hover_above_v1(SIDE, obj_pos.tolist(), clearance=0.15)
print(f"hover above obj (0.15): success={s_hover}, log={log}")

if not s_hover:
    print("First hover failed, trying clearance=0.20")
    s_hover, log = hover_above_v1(SIDE, obj_pos.tolist(), clearance=0.20)
    print(f"hover above obj (0.20): success={s_hover}, log={log}")

if not s_hover:
    print("Second hover failed, trying high waypoint approach")
    high_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.35]
    from scipy.spatial.transform import Rotation as R
    top_down_quat = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat().tolist()
    r_high = freespace_move(right_target_pos=high_pos, right_target_quat=top_down_quat, side=SIDE)
    print(f"High waypoint move: status={r_high.status}")
    if r_high.status == "Success":
        s_hover, log = hover_above_v1(SIDE, obj_pos.tolist(), clearance=0.12)
        print(f"hover after high waypoint (0.12): success={s_hover}, log={log}")

state = get_robot_state()
ee_pos = np.array(state.arms[SIDE].ee_pos)
dist_to_obj_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
print(f"EE after hover: {ee_pos.tolist()}, XY dist to obj: {dist_to_obj_xy:.4f}")

if dist_to_obj_xy > 0.15:
    print(f"WARNING: EE far from object ({dist_to_obj_xy:.3f}m), trying direct move")
    from scipy.spatial.transform import Rotation as R
    top_down_quat = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat().tolist()
    direct_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
    r_direct = freespace_move(right_target_pos=direct_pos, right_target_quat=top_down_quat, side=SIDE)
    print(f"Direct move above obj: status={r_direct.status}")

# Step 3: Refresh object position before grasp
info = get_task_info()
obj_pos = np.array(info["obj_pos"])
original_obj_z = obj_pos[2]
print(f"Refreshed obj_pos: {obj_pos.tolist()}")

# Step 4: Vertical grasp
s_grasp, log = vertical_grasp_v1(SIDE, obj_pos.tolist(), z_offset=grasp_z_offset)
print(f"grasp: success={s_grasp}, log={log}")

# Step 5: Retry with nudge down if grasp failed
retries = 0
while not s_grasp and retries < 4:
    s_grasp, log = nudge_down_and_regrasp_v1(SIDE, delta_z=nudge_delta)
    print(f"regrasp retry {retries}: success={s_grasp}, log={log}")
    retries += 1

# Step 5b: Post-grasp verification using nudge_brutal to escape sink contact.
# nudge_brutal bypasses cuRobo collision checking — needed because after
# descending to grasp depth the arm is often pressed against the sink wall
# and the planner refuses to plan even a small upward move from that state.
grasp_confirmed = False
if s_grasp:
    g_info = get_gripper_info(SIDE)
    print(f"Gripper after grasp: pos={g_info['pos']:.4f}, has_object={g_info['has_object']}, force={g_info.get('actuator_force_N')}")

    # Use nudge_brutal instead of nudge — escapes sink wall contact
    r_test = nudge_brutal(SIDE, delta_pos=[0.0, 0.0, 0.03])
    print(f"Test lift (brutal): success={r_test.success}, final_pos={r_test.final_pos}")

    info_check = get_task_info()
    new_obj_z = info_check["obj_pos"][2]
    obj_rose = (new_obj_z - original_obj_z) > 0.01
    print(f"Object z: original={original_obj_z:.4f}, now={new_obj_z:.4f}, rose={obj_rose}")

    if obj_rose:
        grasp_confirmed = True
    else:
        print("Object did NOT rise - grasp failed physically. Retrying deeper...")
        open_gripper(SIDE)
        # nudge_brutal back down past the object (also collision-blind)
        r_down = nudge_brutal(SIDE, delta_pos=[0.0, 0.0, -0.06])
        print(f"Nudge back down (brutal): success={r_down.success}")

        info = get_task_info()
        obj_pos = np.array(info["obj_pos"])
        close_gripper(SIDE, compliant=True, hold_strength=0.5)
        g_info2 = get_gripper_info(SIDE)
        print(f"Deep re-grasp: pos={g_info2['pos']:.4f}, has_object={g_info2['has_object']}, force={g_info2.get('actuator_force_N')}")

        if g_info2['has_object'] and not g_info2['is_fully_closed']:
            r_test2 = nudge_brutal(SIDE, delta_pos=[0.0, 0.0, 0.04])
            info_check2 = get_task_info()
            new_obj_z2 = info_check2["obj_pos"][2]
            obj_rose2 = (new_obj_z2 - original_obj_z) > 0.01
            print(f"Deep re-grasp test (brutal): obj_z={new_obj_z2:.4f}, rose={obj_rose2}")
            if obj_rose2:
                grasp_confirmed = True

if not grasp_confirmed:
    print("ERROR: All grasp attempts failed — going home.")
    open_gripper(SIDE)
    go_home(SIDE)
    final = get_task_info()
    print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
else:
    # Step 6: Lift using lift_v2 (incremental nudge-based)
    s_lift, log = lift_v2(SIDE, delta_z=0.25)
    print(f"lift: success={s_lift}, log={log}")

    state = get_robot_state()
    ee_z = state.arms[SIDE].ee_pos[2]
    print(f"EE z after lift: {ee_z:.4f}")
    if ee_z < 1.0:
        print("Still low, nudging up more...")
        r_extra = nudge(SIDE, delta_pos=[0.0, 0.0, 0.10])
        print(f"Extra lift: success={r_extra.success}, final_pos={r_extra.final_pos}")

    # Step 7: Hover above place target
    s_ph, log = hover_above_v1(SIDE, place_pos.tolist(), clearance=0.12)
    print(f"hover above place: success={s_ph}, log={log}")

    # Step 8: Place
    s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=0.03)
    print(f"place: success={s_place}, log={log}")

    # Step 9: Retract and go home
    r_up = nudge(SIDE, delta_pos=[0.0, 0.0, 0.15])
    print(f"retract nudge up: success={r_up.success}, final_pos={r_up.final_pos}")

    r_home = go_home(SIDE)
    print(f"go_home: status={r_home.status}")

    final = get_task_info()
    print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
