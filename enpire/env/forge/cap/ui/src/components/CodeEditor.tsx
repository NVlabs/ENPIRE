import Editor from "@monaco-editor/react";

interface CodeEditorProps {
  code: string;
  onChange: (code: string) => void;
  readOnly?: boolean;
  theme?: "dark" | "light";
}

const DEFAULT_CODE = `# CAP Tool API
# Available functions:
#   get_robot_state() -> RobotState
#   set_gripper(side, pos, vel_limit=None, torque_limit=None) -> bool
#   open_gripper(side, vel_limit=None, torque_limit=None) -> bool
#   close_gripper(side, vel_limit=None, torque_limit=None) -> bool
#   grasp(side, vel_limit=10.0) -> dict  # compliant close, stops on contact
#   go_home() -> bool
#   detect_object(query, camera="top", backend="bundlesdf") -> list[Detection3D]
#   freespace_move(left_target_pos, left_target_rpy, right_target_pos, right_target_rpy, ...)
#   nudge(side, delta_pos, delta_rpy) -> NudgeResult
#   execute_skill(skill_name, **params) -> SkillResult
#
# Scene management (sim-only):
#   list_scenes() -> dict          # {ok, scenes, active}
#   setup_scene(name) -> dict      # {ok, scene, objects}
#   clear_table() -> dict          # {ok, removed}

state = get_robot_state()
`;

export default function CodeEditor({ code, onChange, readOnly, theme = "dark" }: CodeEditorProps) {
  return (
    <div className="h-full w-full overflow-hidden rounded-lg border border-base-300">
      <Editor
        height="100%"
        defaultLanguage="python"
        theme={theme === "light" ? "vs" : "vs-dark"}
        value={code || DEFAULT_CODE}
        onChange={(v) => onChange(v ?? "")}
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 13,
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          wordWrap: "on",
          padding: { top: 8 },
          automaticLayout: true,
        }}
      />
    </div>
  );
}
