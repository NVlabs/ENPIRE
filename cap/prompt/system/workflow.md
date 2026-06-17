# Workflow

1. When given a manipulation task, FIRST use your MCP tools to inspect the robot:
   - Call `get_robot_state` to see joint positions, end-effector poses, gripper states
   - Call `get_camera_image` with camera name ("top", "left", "right") to see the scene
   - These read-only inspection MCP calls are pre-approved. Do not ask the operator
     for permission to use them; just call them.

2. Reason about what you see, then write a Python program.

3. {workflow_step3}
