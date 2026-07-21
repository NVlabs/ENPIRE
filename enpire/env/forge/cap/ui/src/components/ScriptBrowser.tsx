import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import * as api from "../api/client";

/** fzf-style fuzzy match: characters must appear in order, case-insensitive. */
function fuzzyMatch(query: string, text: string): { match: boolean; indices: number[] } {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const indices: number[] = [];
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      indices.push(ti);
      qi++;
    }
  }
  return { match: qi === q.length, indices };
}

/** Render text with matched characters highlighted. */
function HighlightedName({ name, indices }: { name: string; indices: number[] }) {
  const set = new Set(indices);
  return (
    <>
      {name.split("").map((ch, i) =>
        set.has(i) ? (
          <span key={i} className="text-primary font-bold">{ch}</span>
        ) : (
          <span key={i}>{ch}</span>
        ),
      )}
    </>
  );
}

function splitScriptPath(path: string): { dir: string; base: string } {
  const idx = path.lastIndexOf("/");
  if (idx < 0) return { dir: "", base: path };
  return { dir: path.slice(0, idx), base: path.slice(idx + 1) };
}

export interface ScriptBrowserHandle {
  focusSearch: () => void;
}

interface ScriptBrowserProps {
  onLoad: (code: string) => void;
  currentCode: string;
}

const ScriptBrowser = forwardRef<ScriptBrowserHandle, ScriptBrowserProps>(function ScriptBrowser({ onLoad, currentCode }, ref) {
  const [scripts, setScripts] = useState<api.ScriptInfo[]>([]);
  const [saveName, setSaveName] = useState("");
  const [renamingIdx, setRenamingIdx] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [showSave, setShowSave] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const [activeScript, setActiveScript] = useState<string | null>(null);
  const activeModifiedRef = useRef<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const listRef = useRef<HTMLDivElement>(null);

  useImperativeHandle(ref, () => ({
    focusSearch: () => searchRef.current?.focus(),
  }));

  const refresh = useCallback(() => {
    api.listScripts().then((list) => {
      setScripts(list);
      // If an active script's modified time changed, re-fetch its content
      if (activeScript) {
        const entry = list.find((s) => s.name === activeScript);
        if (entry && entry.modified !== activeModifiedRef.current) {
          activeModifiedRef.current = entry.modified;
          api.loadScript(entry.name).then((r) => {
            if (r.ok) onLoad(r.code);
          });
        }
      }
    }).catch(() => {});
  }, [activeScript, onLoad]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [refresh]);

  // Keyboard shortcut: Ctrl/Cmd+P to focus search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "p") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Fuzzy-filtered scripts
  const filteredScripts = useMemo(() => {
    if (!searchQuery.trim()) return scripts.map((s) => ({ script: s, indices: [] as number[] }));
    return scripts
      .map((s) => {
        const { match, indices } = fuzzyMatch(searchQuery, s.name);
        return { script: s, indices, match };
      })
      .filter((r) => r.match);
  }, [scripts, searchQuery]);

  // Reset selection when filtered list changes
  useEffect(() => {
    setSelectedIdx(-1);
  }, [filteredScripts.length, searchQuery]);

  // Scroll selected item into view
  useEffect(() => {
    if (selectedIdx < 0 || !listRef.current) return;
    const rows = listRef.current.querySelectorAll("[data-script-row]");
    rows[selectedIdx]?.scrollIntoView({ block: "nearest" });
  }, [selectedIdx]);

  const handleSave = async () => {
    const name = saveName.trim();
    if (!name || !currentCode.trim()) return;
    await api.saveScript(name, currentCode);
    setSaveName("");
    setShowSave(false);
    refresh();
  };

  const handleLoad = async (name: string) => {
    const result = await api.loadScript(name);
    if (result.ok) {
      setActiveScript(name);
      const entry = scripts.find((s) => s.name === name);
      activeModifiedRef.current = entry?.modified ?? null;
      onLoad(result.code);
    }
  };

  const handleDelete = async (name: string) => {
    await api.deleteScript(name);
    refresh();
  };

  const handleRename = async (oldName: string) => {
    const newName = renameValue.trim();
    if (!newName || newName === oldName) {
      setRenamingIdx(null);
      return;
    }
    await api.renameScript(oldName, newName);
    setRenamingIdx(null);
    setRenameValue("");
    refresh();
  };

  return (
    <div className="flex h-full flex-col bg-base-200">
      <div className="p-2 space-y-1 shrink-0">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm">Saved Scripts</h3>
          <button
            className="btn btn-primary btn-xs"
            onClick={() => setShowSave(!showSave)}
          >
            {showSave ? "Cancel" : "Save Current"}
          </button>
        </div>

        {/* Fuzzy search box */}
        <div className="relative">
          <input
            ref={searchRef}
            type="text"
            className="input input-bordered input-xs w-full pl-7"
            placeholder="Search scripts... (Ctrl+P)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                if (selectedIdx >= 0) {
                  setSelectedIdx(-1);
                } else {
                  setSearchQuery("");
                  searchRef.current?.blur();
                }
              } else if (e.key === "ArrowDown") {
                e.preventDefault();
                setSelectedIdx((prev) =>
                  prev < filteredScripts.length - 1 ? prev + 1 : prev,
                );
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setSelectedIdx((prev) => (prev > 0 ? prev - 1 : prev));
                } else if (e.key === "Enter") {
                  e.preventDefault();
                  if (selectedIdx >= 0 && selectedIdx < filteredScripts.length) {
                  const selected = filteredScripts[selectedIdx];
                  if (!selected) return;
                  handleLoad(selected.script.name);
                  setSelectedIdx(-1);
                } else if (filteredScripts.length > 0) {
                  // First Enter highlights first item, second Enter loads it
                  setSelectedIdx(0);
                }
              }
            }}
          />
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-base-content/40"
          >
            <path
              fillRule="evenodd"
              d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
              clipRule="evenodd"
            />
          </svg>
        </div>

        {/* Save dialog */}
        {showSave && (
          <div className="flex gap-2">
            <input
              type="text"
              className="input input-bordered input-xs flex-1"
              placeholder="Script path (e.g. peg/recover)..."
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
            />
            <button
              className="btn btn-success btn-xs"
              onClick={handleSave}
              disabled={!saveName.trim() || !currentCode.trim()}
            >
              Save
            </button>
          </div>
        )}
      </div>

      {/* Script list */}
      <div ref={listRef} className="flex-1 min-h-0 overflow-y-auto px-2 pb-2">
        {filteredScripts.length === 0 ? (
          <div className="text-base-content/30 text-xs py-2 text-center">
            {searchQuery ? "No matches" : "No saved scripts"}
          </div>
        ) : (
          <table className="table table-xs w-full">
            <tbody>
              {filteredScripts.map(({ script: s, indices }, i) => (
                <tr
                  key={s.name}
                  data-script-row
                  className={`hover cursor-pointer transition-colors ${
                    i === selectedIdx
                      ? "bg-primary/15 outline outline-1 outline-primary/30"
                      : ""
                  } ${activeScript === s.name ? "bg-base-300" : ""}`}
                  onClick={() => handleLoad(s.name)}
                >
                  <td className="font-mono text-xs">
                    {renamingIdx === i ? (
                      <input
                        type="text"
                        className="input input-bordered input-xs w-full"
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleRename(s.name);
                          if (e.key === "Escape") setRenamingIdx(null);
                        }}
                        onBlur={() => handleRename(s.name)}
                        autoFocus
                      />
                    ) : (
                      <span
                        className="cursor-pointer hover:text-primary"
                        title="Click to load"
                      >
                        {searchQuery ? (
                          <HighlightedName name={s.name} indices={indices} />
                        ) : (
                          (() => {
                            const { dir, base } = splitScriptPath(s.name);
                            return (
                              <span className="block min-w-0">
                                <span className="block truncate">{base}</span>
                                {dir ? (
                                  <span className="block truncate text-[10px] text-base-content/50">
                                    {dir}
                                  </span>
                                ) : null}
                              </span>
                            );
                          })()
                        )}
                      </span>
                    )}
                  </td>
                  <td className="text-xs text-base-content/40 w-24">
                    {s.modified}
                  </td>
                  <td className="w-20 text-right">
                    <button
                      className="btn btn-ghost btn-xs"
                      onClick={(e) => {
                        e.stopPropagation();
                        setRenamingIdx(i);
                        setRenameValue(s.name);
                      }}
                      title="Rename"
                    >
                      Rn
                    </button>
                    <button
                      className="btn btn-ghost btn-xs text-error"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(s.name);
                      }}
                      title="Delete"
                    >
                      X
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
});

export default ScriptBrowser;
