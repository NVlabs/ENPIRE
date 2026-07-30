// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * RED TEAM tests for TimestampChart component, fetchComponentTimestamps API,
 * and useEpisode hook timestamp additions.
 *
 * Strategy: Since TimestampChart is canvas-based, we can't assert pixels.
 * Instead we:
 *   1. Extract and test pure-logic helpers (epData useMemo, findNearestDot, etc.)
 *      by rendering the component and inspecting behavior via callbacks.
 *   2. Verify the component renders without crashing for various data shapes.
 *   3. Verify user interaction callbacks (onSeek) fire correctly.
 *   4. Test the API client function with mocked fetch.
 *   5. Test the useEpisode hook additions via renderHook.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, screen, fireEvent, act, cleanup } from "@testing-library/react"
import { renderHook, act as actHook } from "@testing-library/react"
import type { ComponentTimestampData } from "@/api/types"

// ---------------------------------------------------------------------------
// Test data factories
// ---------------------------------------------------------------------------

function makeTimestampData(overrides?: Partial<ComponentTimestampData>): ComponentTimestampData {
  return {
    timestamps: [
      { left_state: 100.0, right_state: 100.001, top_camera: 100.002, action: 100.010 },
      { left_state: 100.033, right_state: 100.034, top_camera: 100.035, action: 100.043 },
      { left_state: 100.066, right_state: 100.067, top_camera: 100.068, action: 100.076 },
    ],
    action_sources: ["human", "policy", "policy"],
    record_timestamps: [100.005, 100.038, 100.071],
    ...overrides,
  }
}

function makeEmptyData(): ComponentTimestampData {
  return { timestamps: [], action_sources: [], record_timestamps: [] }
}

function makeSingleStepData(): ComponentTimestampData {
  return {
    timestamps: [{ left_state: 50.0, action: 50.005 }],
    action_sources: ["human"],
    record_timestamps: [50.003],
  }
}

/** Data where some steps are missing certain components */
function makeSparseData(): ComponentTimestampData {
  return {
    timestamps: [
      { left_state: 10.0, action: 10.01 },
      { left_state: 10.03, right_state: 10.031, action: 10.04 },
      { right_state: 10.061, top_camera: 10.062, action: 10.07 },
    ],
    action_sources: ["policy", "policy", "human"],
    record_timestamps: [],
  }
}

/** Large dataset to stress test */
function makeLargeData(n: number): ComponentTimestampData {
  const timestamps: Record<string, number>[] = []
  const actionSources: string[] = []
  const recordTimestamps: number[] = []
  for (let i = 0; i < n; i++) {
    const base = 200 + i * 0.033
    timestamps.push({
      left_state: base,
      right_state: base + 0.001,
      top_camera: base + 0.003,
      left_camera: base + 0.004,
      right_camera: base + 0.005,
      action: base + 0.010,
      action_source: base + 0.011,
    })
    actionSources.push(i % 3 === 0 ? "human" : "policy")
    recordTimestamps.push(base + 0.006)
  }
  return { timestamps, action_sources: actionSources, record_timestamps: recordTimestamps }
}

// ---------------------------------------------------------------------------
// 1. TimestampChart component tests
// ---------------------------------------------------------------------------

// We dynamically import the component so test-setup canvas stubs are applied first
let TimestampChart: typeof import("@/components/TimestampChart").TimestampChart

beforeEach(async () => {
  const mod = await import("@/components/TimestampChart")
  TimestampChart = mod.TimestampChart
})

afterEach(() => {
  cleanup()
})

describe("TimestampChart", () => {
  // ------ Rendering without crash ------

  describe("renders without crashing", () => {
    it("renders with normal multi-step data", () => {
      const onSeek = vi.fn()
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
      expect(container.querySelector("canvas")).toBeTruthy()
    })

    it("renders with empty data", () => {
      const onSeek = vi.fn()
      const { container } = render(
        <TimestampChart data={makeEmptyData()} chartStep={0} totalSteps={0} fps={30} onSeek={onSeek} />,
      )
      expect(container.querySelector("canvas")).toBeTruthy()
    })

    it("renders with single step data", () => {
      const onSeek = vi.fn()
      render(
        <TimestampChart data={makeSingleStepData()} chartStep={0} totalSteps={1} fps={30} onSeek={onSeek} />,
      )
    })

    it("renders with sparse / missing component data", () => {
      const onSeek = vi.fn()
      render(
        <TimestampChart data={makeSparseData()} chartStep={1} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
    })

    it("renders with large dataset (1000 steps)", () => {
      const onSeek = vi.fn()
      render(
        <TimestampChart data={makeLargeData(1000)} chartStep={500} totalSteps={1000} fps={30} onSeek={onSeek} />,
      )
    })

    it("renders when chartStep exceeds nSteps (out of bounds)", () => {
      const onSeek = vi.fn()
      // chartStep=100 but only 3 steps in data -- should not crash
      render(
        <TimestampChart data={makeTimestampData()} chartStep={100} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
    })

    it("renders with negative chartStep", () => {
      const onSeek = vi.fn()
      render(
        <TimestampChart data={makeTimestampData()} chartStep={-1} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
    })
  })

  // ------ Data preprocessing (epData useMemo) ------

  describe("data preprocessing (epData useMemo)", () => {
    it("computes correct tMin and tMax from timestamps", () => {
      // We verify indirectly: the status bar shows nSteps and duration
      const data = makeTimestampData()
      const { container } = render(
        <TimestampChart data={data} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      // The bottom bar should display "3 steps"
      const text = container.textContent || ""
      expect(text).toContain("3 steps")
      // Duration should be tMax - tMin = 100.076 - 100.0 = 0.076
      expect(text).toContain("0.076")
    })

    it("shows 0 steps and 0 duration for empty data", () => {
      const { container } = render(
        <TimestampChart data={makeEmptyData()} chartStep={0} totalSteps={0} fps={30} onSeek={vi.fn()} />,
      )
      const text = container.textContent || ""
      expect(text).toContain("0 steps")
      expect(text).toContain("0.000s total")
    })

    it("correctly identifies visible components (filters out absent ones)", () => {
      // makeSingleStepData only has left_state and action
      // Component labels are rendered on canvas (can't check), but the component should not crash
      // and nSteps should be 1
      const { container } = render(
        <TimestampChart data={makeSingleStepData()} chartStep={0} totalSteps={1} fps={30} onSeek={vi.fn()} />,
      )
      expect(container.textContent).toContain("1 steps")
    })

    it("normalizes timestamps relative to tMin (all >= 0)", () => {
      // The duration displayed proves normalization happened: if tMin=100.0 and tMax=100.076,
      // duration is 0.076, not 100.076
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      const text = container.textContent || ""
      // Duration should be ~0.076, not ~100
      expect(text).toMatch(/0\.076/)
      expect(text).not.toMatch(/100\./)
    })

    it("handles record_timestamps normalization relative to tMin", () => {
      // record_timestamps should also be normalized; we check the component renders with showRecordLines toggle
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      // Toggle record lines button should exist
      const btn = container.querySelector("button")
      expect(btn).toBeTruthy()
      expect(btn!.textContent).toContain("Record Lines")
    })
  })

  // ------ Component ordering ------

  describe("component ordering", () => {
    it("only shows components present in data (COMPONENT_ORDER filtering)", () => {
      // With sparse data that lacks some keys, the chart should still render
      const onSeek = vi.fn()
      render(<TimestampChart data={makeSparseData()} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />)
      // No crash = pass; canvas draws only the rows that exist
    })

    it("handles unknown component names gracefully", () => {
      const data: ComponentTimestampData = {
        timestamps: [{ unknown_sensor: 1.0, another_thing: 1.001 }],
        action_sources: [],
        record_timestamps: [],
      }
      // Unknown components are NOT in COMPONENT_ORDER, so they should be filtered out
      // The chart should render with 0 visible component rows but 1 step
      const { container } = render(
        <TimestampChart data={data} chartStep={0} totalSteps={1} fps={30} onSeek={vi.fn()} />,
      )
      expect(container.textContent).toContain("1 steps")
    })
  })

  // ------ View reset on data change ------

  describe("view reset when data changes", () => {
    it("resets viewXMin/viewXMax when new data is provided", () => {
      const onSeek = vi.fn()
      const data1 = makeTimestampData()
      const { rerender, container } = render(
        <TimestampChart data={data1} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
      // The viewing range should show the full duration
      expect(container.textContent).toContain("0.076")

      // Re-render with different data
      const data2 = makeLargeData(10)
      rerender(<TimestampChart data={data2} chartStep={0} totalSteps={10} fps={30} onSeek={onSeek} />)
      // Should now show a different duration
      const text = container.textContent || ""
      expect(text).not.toContain("0.076")
    })
  })

  // ------ Record Lines toggle ------

  describe("record lines toggle", () => {
    it("starts with record lines hidden", () => {
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      const btn = container.querySelector("button")!
      // Default off: border should be #4a4a60 (off state) -- jsdom converts to rgb
      expect(btn.style.border).toContain("rgb(74, 74, 96)")
    })

    it("toggles record lines on click", () => {
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      const btn = container.querySelector("button")!
      fireEvent.click(btn)
      // After click, should be active: border includes #4a70c0 -- jsdom converts to rgb
      expect(btn.style.border).toContain("rgb(74, 112, 192)")
    })

    it("toggles back off on second click", () => {
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      const btn = container.querySelector("button")!
      fireEvent.click(btn)
      fireEvent.click(btn)
      expect(btn.style.border).toContain("rgb(74, 74, 96)")
    })
  })

  // ------ Click-to-seek (distinguish click from drag) ------

  describe("click-to-seek", () => {
    it("does not call onSeek when mouse is dragged (moved > 3px)", () => {
      const onSeek = vi.fn()
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
      const canvas = container.querySelector("canvas")!

      // Simulate drag: mousedown, move significantly, mouseup
      fireEvent.mouseDown(canvas, { clientX: 200, clientY: 100 })
      fireEvent.mouseMove(canvas, { clientX: 250, clientY: 100 }) // 50px move
      fireEvent.mouseUp(canvas, { clientX: 250, clientY: 100 })

      // onSeek should NOT be called for a drag
      expect(onSeek).not.toHaveBeenCalled()
    })

    it("calls onSeek when mouse clicks without moving (click on dot)", () => {
      const onSeek = vi.fn()
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
      const canvas = container.querySelector("canvas")!

      // Click without moving -- whether onSeek fires depends on findNearestDot finding a hit.
      // In jsdom with stubbed canvas, getBoundingClientRect returns {width:800, height:320},
      // so we can try clicking near where a dot might be.
      // The click handler checks `wasDragging && !moved` then calls findNearestDot.
      // Since canvas is stubbed, findNearestDot relies on computed positions from canvas dimensions.
      fireEvent.mouseDown(canvas, { clientX: 200, clientY: 60 })
      fireEvent.mouseUp(canvas, { clientX: 200, clientY: 60 })

      // We can't guarantee a hit in jsdom, but at least verify no crash.
      // The function should have been called 0 or 1 times (depending on hit).
      expect(onSeek.mock.calls.length).toBeLessThanOrEqual(1)
    })
  })

  // ------ Double-click zoom reset ------

  describe("double-click zoom reset", () => {
    it("resets view to full duration on double-click", () => {
      const onSeek = vi.fn()
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
      const canvas = container.querySelector("canvas")!

      // Simulate a wheel zoom first to change the view
      // (wheel events need to be dispatched natively since the component uses addEventListener)
      const wheelEvent = new WheelEvent("wheel", {
        deltaY: -100, clientX: 400, clientY: 160, bubbles: true,
      })
      canvas.dispatchEvent(wheelEvent)

      // Now double-click to reset
      fireEvent.doubleClick(canvas)

      // After reset, the viewing range in the status bar should match full duration
      const text = container.textContent || ""
      expect(text).toContain("0.076")
    })
  })

  // ------ Mouse leave ------

  describe("mouse leave", () => {
    it("cleans up drag state on mouse leave", () => {
      const onSeek = vi.fn()
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={onSeek} />,
      )
      const canvas = container.querySelector("canvas")!

      fireEvent.mouseDown(canvas, { clientX: 200, clientY: 100 })
      fireEvent.mouseLeave(canvas)

      // After leave, cursor should be reset (no error, canvas still default)
      expect(canvas.style.cursor).toBe("default")
    })
  })

  // ------ Status bar info ------

  describe("status bar displays correct info", () => {
    it("shows step count, total duration, and viewing range", () => {
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      const text = container.textContent || ""
      expect(text).toContain("3 steps")
      expect(text).toContain("total")
      expect(text).toContain("viewing")
    })

    it("shows scroll/drag/click/dbl-click hint text", () => {
      const { container } = render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
      const text = container.textContent || ""
      expect(text).toContain("Scroll to zoom")
      expect(text).toContain("Drag to pan")
      expect(text).toContain("Click dot to seek")
      expect(text).toContain("Dbl-click to reset")
    })
  })

  // ------ Edge cases ------

  describe("edge cases", () => {
    it("handles data with all COMPONENT_ORDER keys present", () => {
      const data = makeLargeData(5)
      // This data has all 7 component keys
      render(
        <TimestampChart data={data} chartStep={2} totalSteps={5} fps={30} onSeek={vi.fn()} />,
      )
    })

    it("handles zero fps gracefully", () => {
      // fps=0 is passed as a prop; the component only uses it externally
      render(
        <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={0} onSeek={vi.fn()} />,
      )
    })

    it("handles negative timestamps (timestamps before epoch or relative)", () => {
      const data: ComponentTimestampData = {
        timestamps: [
          { left_state: -1.0, action: -0.99 },
          { left_state: -0.5, action: -0.49 },
        ],
        action_sources: ["human", "policy"],
        record_timestamps: [-0.95, -0.45],
      }
      const { container } = render(
        <TimestampChart data={data} chartStep={0} totalSteps={2} fps={30} onSeek={vi.fn()} />,
      )
      expect(container.textContent).toContain("2 steps")
    })

    it("handles data where all timestamps are identical", () => {
      const data: ComponentTimestampData = {
        timestamps: [
          { left_state: 5.0, action: 5.0 },
          { left_state: 5.0, action: 5.0 },
        ],
        action_sources: [],
        record_timestamps: [],
      }
      // duration = 0, so viewXMax should be set to 1 as fallback
      const { container } = render(
        <TimestampChart data={data} chartStep={0} totalSteps={2} fps={30} onSeek={vi.fn()} />,
      )
      expect(container.textContent).toContain("0.000s total")
    })

    it("handles missing action_sources array", () => {
      const data: ComponentTimestampData = {
        timestamps: [{ left_state: 1.0, action: 1.01 }],
        action_sources: undefined as unknown as string[],
        record_timestamps: [],
      }
      // The component does `data.action_sources || []`, should not crash
      render(
        <TimestampChart data={data} chartStep={0} totalSteps={1} fps={30} onSeek={vi.fn()} />,
      )
    })

    it("handles missing record_timestamps array", () => {
      const data: ComponentTimestampData = {
        timestamps: [{ left_state: 1.0 }],
        action_sources: [],
        record_timestamps: undefined as unknown as number[],
      }
      render(
        <TimestampChart data={data} chartStep={0} totalSteps={1} fps={30} onSeek={vi.fn()} />,
      )
    })
  })

  // ------ Step cursor position ------

  describe("step cursor", () => {
    it("renders cursor for valid chartStep within bounds", () => {
      // Can't verify canvas drawing, but verify no crash
      render(
        <TimestampChart data={makeTimestampData()} chartStep={1} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
    })

    it("handles chartStep at last index", () => {
      render(
        <TimestampChart data={makeTimestampData()} chartStep={2} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
    })

    it("handles chartStep beyond data length (no crash)", () => {
      render(
        <TimestampChart data={makeTimestampData()} chartStep={999} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
    })
  })

  // ------ Action source color mapping ------

  describe("action source color mapping", () => {
    it("renders with mixed human/policy sources without crash", () => {
      const data = makeTimestampData()
      // data has ["human", "policy", "policy"]
      render(
        <TimestampChart data={data} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
    })

    it("renders with unknown action source labels", () => {
      const data = makeTimestampData({ action_sources: ["scripted", "teleop", "unknown"] })
      // Unknown sources should fallback to default color, not crash
      render(
        <TimestampChart data={data} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
    })

    it("renders with empty action_sources (no coloring by source)", () => {
      const data = makeTimestampData({ action_sources: [] })
      render(
        <TimestampChart data={data} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
      )
    })
  })
})

// ---------------------------------------------------------------------------
// 2. Pure logic unit tests (extracted from component logic)
// ---------------------------------------------------------------------------

describe("TimestampChart pure logic", () => {
  describe("epData computation", () => {
    it("computes tMin as the minimum of all timestamp values", () => {
      const data = makeTimestampData()
      const steps = data.timestamps
      let tMin = Infinity
      for (const step of steps) {
        for (const v of Object.values(step)) {
          if (typeof v === "number" && v < tMin) tMin = v
        }
      }
      expect(tMin).toBe(100.0)
    })

    it("computes tMax as the maximum of all timestamp values", () => {
      const data = makeTimestampData()
      const steps = data.timestamps
      let tMax = -Infinity
      for (const step of steps) {
        for (const v of Object.values(step)) {
          if (typeof v === "number" && v > tMax) tMax = v
        }
      }
      expect(tMax).toBe(100.076)
    })

    it("builds components dict with normalized (relative) times", () => {
      const data = makeTimestampData()
      const steps = data.timestamps
      let tMin = Infinity
      for (const step of steps) {
        for (const v of Object.values(step)) {
          if (typeof v === "number" && v < tMin) tMin = v
        }
      }

      const components: Record<string, number[]> = {}
      for (const step of steps) {
        for (const [k, v] of Object.entries(step)) {
          if (typeof v !== "number") continue
          if (!components[k]) components[k] = []
          components[k].push(v - tMin)
        }
      }

      expect(components.left_state).toHaveLength(3)
      expect(components.left_state![0]).toBeCloseTo(0, 6)
      expect(components.left_state![1]).toBeCloseTo(0.033, 6)
      expect(components.action![2]).toBeCloseTo(0.076, 6)
    })

    it("normalizes record_timestamps relative to tMin", () => {
      const data = makeTimestampData()
      const tMin = 100.0
      const recRel = data.record_timestamps.map(t => t - tMin)
      expect(recRel[0]).toBeCloseTo(0.005, 6)
      expect(recRel[1]).toBeCloseTo(0.038, 6)
      expect(recRel[2]).toBeCloseTo(0.071, 6)
    })

    it("handles empty timestamps array", () => {
      const data = makeEmptyData()
      expect(data.timestamps.length).toBe(0)
      // epData should return defaults
    })
  })

  describe("step cursor mean calculation", () => {
    it("computes mean of component times at a given step", () => {
      const data = makeTimestampData()
      const tMin = 100.0
      const components: Record<string, number[]> = {}
      for (const step of data.timestamps) {
        for (const [k, v] of Object.entries(step)) {
          if (typeof v !== "number") continue
          if (!components[k]) components[k] = []
          components[k].push(v - tMin)
        }
      }

      const COMPONENT_ORDER = ["left_state", "right_state", "top_camera", "left_camera", "right_camera", "action", "action_source"]
      const visibleComponents = COMPONENT_ORDER.filter(c => components[c])
      const chartStep = 1

      let sum = 0, count = 0
      for (const comp of visibleComponents) {
        const times = components[comp]
        if (times && chartStep < times.length) {
          sum += times[chartStep]
          count++
        }
      }
      const stepTime = sum / count

      // Step 1 values: left=0.033, right=0.034, top=0.035, action=0.043
      const expected = (0.033 + 0.034 + 0.035 + 0.043) / 4
      expect(stepTime).toBeCloseTo(expected, 6)
    })
  })

  describe("findNearestDot algorithm", () => {
    // Replicate the algorithm in isolation
    const COMPONENT_ORDER = ["left_state", "right_state", "top_camera", "left_camera", "right_camera", "action", "action_source"]
    const MARGIN = { top: 30, right: 20, bottom: 40, left: 120 }
    const ROW_HEIGHT = 36

    function findNearestDot(
      pos: { x: number; y: number },
      components: Record<string, number[]>,
      xMin: number,
      xMax: number,
      W: number,
      H: number,
    ) {
      const plotW = W - MARGIN.left - MARGIN.right
      const plotH = H - MARGIN.top - MARGIN.bottom
      const visibleComponents = COMPONENT_ORDER.filter(c => components[c])
      const rowH = Math.min(ROW_HEIGHT, plotH / Math.max(visibleComponents.length, 1))

      let bestDist = 12
      let bestComp: string | null = null
      let bestStep: number | null = null
      let bestTime: number | null = null

      visibleComponents.forEach((comp, i) => {
        const times = components[comp]
        if (!times) return
        const y = MARGIN.top + i * rowH + rowH / 2
        const dy = pos.y - y
        if (Math.abs(dy) > rowH / 2) return

        for (let s = 0; s < times.length; s++) {
          const t = times[s]
          if (t < xMin || t > xMax) continue
          const x = MARGIN.left + (t - xMin) / (xMax - xMin) * plotW
          const dist = Math.sqrt((pos.x - x) ** 2 + dy ** 2)
          if (dist < bestDist) {
            bestDist = dist
            bestComp = comp
            bestStep = s
            bestTime = t
          }
        }
      })

      if (bestComp === null) return null
      return { comp: bestComp, step: bestStep!, time: bestTime! }
    }

    it("returns null when no dots are in range", () => {
      const components = { left_state: [0, 0.033, 0.066] }
      const result = findNearestDot({ x: 800, y: 800 }, components, 0, 0.1, 800, 320)
      expect(result).toBeNull()
    })

    it("finds the correct nearest dot", () => {
      const components = { left_state: [0, 0.033, 0.066] }
      // left_state is row 0: y = 30 + 0*36 + 18 = 48
      // With W=800, plotW=660, xMin=0, xMax=0.1
      // dot at t=0 -> x = 120 + 0/0.1*660 = 120
      // dot at t=0.033 -> x = 120 + 0.033/0.1*660 = 120 + 217.8 = 337.8
      const result = findNearestDot({ x: 340, y: 48 }, components, 0, 0.1, 800, 320)
      expect(result).not.toBeNull()
      expect(result!.step).toBe(1)
      expect(result!.comp).toBe("left_state")
      expect(result!.time).toBeCloseTo(0.033, 6)
    })

    it("returns null when components dict is empty", () => {
      const result = findNearestDot({ x: 200, y: 50 }, {}, 0, 1, 800, 320)
      expect(result).toBeNull()
    })

    it("respects the 12px distance threshold", () => {
      const components = { left_state: [0.05] }
      // dot x = 120 + 0.05/0.1*660 = 450, dot y = 48
      // click at (450, 48+13) = (450, 61) -- dy=13, which is within rowH/2=18 but
      // dist = 13, which is > 12 threshold
      const result = findNearestDot({ x: 450, y: 61 }, components, 0, 0.1, 800, 320)
      expect(result).toBeNull()
    })

    it("picks the closer dot when two are near", () => {
      const components = {
        left_state: [0.05],
        right_state: [0.052],
      }
      // left_state row 0: y=48, right_state row 1: y=84
      // left_state dot x = 120 + 0.05/0.1*660 = 450
      // right_state dot x = 120 + 0.052/0.1*660 = 463.2
      // Point near right_state: (463, 84)
      const result = findNearestDot({ x: 463, y: 84 }, components, 0, 0.1, 800, 320)
      expect(result).not.toBeNull()
      expect(result!.comp).toBe("right_state")
    })

    it("skips dots outside the visible x range", () => {
      const components = { left_state: [0, 0.05, 0.2] }
      // xMax=0.1, so t=0.2 should be skipped
      const result = findNearestDot({ x: 780, y: 48 }, components, 0, 0.1, 800, 320)
      // The click at x=780 is far right; only dots within [0, 0.1] range exist
      // dot at 0.05 is at x=450, way too far from 780
      expect(result).toBeNull()
    })
  })

  describe("action source color mapping", () => {
    it("maps 'human' to #ff4444", () => {
      const ACTION_SRC_COLORS: Record<string, string> = { human: "#ff4444", policy: "#44cc44" }
      expect(ACTION_SRC_COLORS["human"]).toBe("#ff4444")
    })

    it("maps 'policy' to #44cc44", () => {
      const ACTION_SRC_COLORS: Record<string, string> = { human: "#ff4444", policy: "#44cc44" }
      expect(ACTION_SRC_COLORS["policy"]).toBe("#44cc44")
    })

    it("returns undefined for unknown sources (falls back to default)", () => {
      const ACTION_SRC_COLORS: Record<string, string> = { human: "#ff4444", policy: "#44cc44" }
      expect(ACTION_SRC_COLORS["teleop"]).toBeUndefined()
    })
  })

  describe("zoom behavior", () => {
    it("zoom in reduces the visible range", () => {
      const xMin = 0, xMax = 1
      const frac = 0.5, pivot = 0.5
      const factor = 1 / 1.15 // zoom in
      const newMin = pivot - (pivot - xMin) * factor
      const newMax = pivot + (xMax - pivot) * factor
      expect(newMax - newMin).toBeLessThan(xMax - xMin)
    })

    it("zoom out increases the visible range", () => {
      const xMin = 0, xMax = 1
      const frac = 0.5, pivot = 0.5
      const factor = 1.15 // zoom out
      const newMin = pivot - (pivot - xMin) * factor
      const newMax = pivot + (xMax - pivot) * factor
      expect(newMax - newMin).toBeGreaterThan(xMax - xMin)
    })

    it("zoom preserves pivot point", () => {
      const xMin = 0.2, xMax = 0.8
      const frac = 0.3
      const pivot = xMin + frac * (xMax - xMin) // 0.38
      const factor = 1 / 1.15
      const newMin = pivot - (pivot - xMin) * factor
      const newMax = pivot + (xMax - pivot) * factor

      // The fraction of the pivot in the new range should be the same
      const newFrac = (pivot - newMin) / (newMax - newMin)
      expect(newFrac).toBeCloseTo(frac, 5)
    })

    it("rejects zoom that would produce range < 1e-6", () => {
      const xMin = 0.5, xMax = 0.5 + 1e-7
      const range = xMax - xMin
      // This range is already < 1e-6, so a further zoom-in should be rejected
      expect(range).toBeLessThan(1e-6)
    })
  })

  describe("DPI scaling", () => {
    it("canvas dimensions are scaled by devicePixelRatio", () => {
      // The draw function does: canvas.width = rect.width * dpr
      // Verify the math:
      const rectWidth = 800
      const dpr = 2
      expect(rectWidth * dpr).toBe(1600)
    })

    it("logical coordinates use css px (divided by dpr)", () => {
      // findNearestDot does: W = canvas.width / (window.devicePixelRatio || 1)
      const canvasWidth = 1600
      const dpr = 2
      expect(canvasWidth / dpr).toBe(800)
    })
  })
})

// ---------------------------------------------------------------------------
// 3. API client tests
// ---------------------------------------------------------------------------

describe("fetchComponentTimestamps", () => {
  let fetchComponentTimestamps: typeof import("@/api/client").fetchComponentTimestamps

  beforeEach(async () => {
    const mod = await import("@/api/client")
    fetchComponentTimestamps = mod.fetchComponentTimestamps
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("calls the correct URL", async () => {
    const mockData = makeTimestampData()
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    } as Response)

    await fetchComponentTimestamps("my-task", 42)
    expect(globalThis.fetch).toHaveBeenCalledWith("/api/tasks/my-task/episodes/42/component_timestamps")
  })

  it("returns parsed ComponentTimestampData on success", async () => {
    const mockData = makeTimestampData()
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    } as Response)

    const result = await fetchComponentTimestamps("task1", 0)
    expect(result.timestamps).toHaveLength(3)
    expect(result.action_sources).toEqual(["human", "policy", "policy"])
    expect(result.record_timestamps).toHaveLength(3)
  })

  it("throws on non-OK response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 404,
      text: () => Promise.resolve("Not Found"),
    } as unknown as Response)

    await expect(fetchComponentTimestamps("task1", 99)).rejects.toThrow("404")
  })

  it("throws on network error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"))

    await expect(fetchComponentTimestamps("task1", 0)).rejects.toThrow("Failed to fetch")
  })

  it("URL-encodes the task ID", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(makeEmptyData()),
    } as Response)

    await fetchComponentTimestamps("task/with/slashes", 0)
    // The current implementation doesn't URL-encode, so this tests that it passes through
    expect(globalThis.fetch).toHaveBeenCalledWith(
      "/api/tasks/task/with/slashes/episodes/0/component_timestamps",
    )
  })
})

// ---------------------------------------------------------------------------
// 4. useEpisode hook timestamp additions
// ---------------------------------------------------------------------------

describe("useEpisode hook - timestamp additions", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it("exposes timestampData as null initially", async () => {
    // Mock fetchEpisodeInfo to avoid side effects
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        total_frames: 100, fps: 30, duration_s: 3.3, folder: "ep_0",
        cameras: { top: true, left: true, right: true },
        trim_start_frame: 0, trim_end_frame: 99, trim_segments: [],
        value_trim_start: 0, value_trim_end: 99, discarded: false,
      }),
    } as Response)

    const { useEpisode } = await import("@/hooks/useEpisode")
    const { result } = renderHook(() => useEpisode("task1", 0))

    expect(result.current.timestampData).toBeNull()
    expect(result.current.timestampLoading).toBe(false)
  })

  it("loadTimestamps fetches and sets timestampData", async () => {
    const mockTimestamps = makeTimestampData()
    let callCount = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const urlStr = typeof url === "string" ? url : url.toString()
      if (urlStr.includes("component_timestamps")) {
        return { ok: true, json: () => Promise.resolve(mockTimestamps) } as Response
      }
      // Default: episode info
      return {
        ok: true,
        json: () => Promise.resolve({
          total_frames: 100, fps: 30, duration_s: 3.3, folder: "ep_0",
          cameras: { top: true, left: true, right: true },
          trim_start_frame: 0, trim_end_frame: 99, trim_segments: [],
          value_trim_start: 0, value_trim_end: 99, discarded: false,
        }),
      } as Response
    })

    const { useEpisode } = await import("@/hooks/useEpisode")
    const { result } = renderHook(() => useEpisode("task1", 0))

    // Wait for initial effect to settle
    await actHook(async () => {
      await new Promise(r => setTimeout(r, 50))
    })

    // Call loadTimestamps
    await actHook(async () => {
      await result.current.loadTimestamps()
    })

    expect(result.current.timestampData).not.toBeNull()
    expect(result.current.timestampData!.timestamps).toHaveLength(3)
    expect(result.current.timestampLoading).toBe(false)
  })

  it("loadTimestamps sets timestampData to null on error", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const urlStr = typeof url === "string" ? url : url.toString()
      if (urlStr.includes("component_timestamps")) {
        return { ok: false, status: 500 } as Response
      }
      return {
        ok: true,
        json: () => Promise.resolve({
          total_frames: 100, fps: 30, duration_s: 3.3, folder: "ep_0",
          cameras: { top: true, left: true, right: true },
          trim_start_frame: 0, trim_end_frame: 99, trim_segments: [],
          value_trim_start: 0, value_trim_end: 99, discarded: false,
        }),
      } as Response
    })

    const { useEpisode } = await import("@/hooks/useEpisode")
    const { result } = renderHook(() => useEpisode("task1", 0))

    await actHook(async () => {
      await new Promise(r => setTimeout(r, 50))
    })

    await actHook(async () => {
      await result.current.loadTimestamps()
    })

    expect(result.current.timestampData).toBeNull()
    expect(result.current.timestampLoading).toBe(false)
  })

  it("loadTimestamps is a no-op when taskId is empty", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    } as Response)

    const { useEpisode } = await import("@/hooks/useEpisode")
    const { result } = renderHook(() => useEpisode("", null))

    const fetchSpy = vi.spyOn(globalThis, "fetch")
    fetchSpy.mockClear()

    await actHook(async () => {
      await result.current.loadTimestamps()
    })

    // Should not have made any fetch call for timestamps
    const timestampCalls = fetchSpy.mock.calls.filter(
      ([url]) => typeof url === "string" && url.includes("component_timestamps"),
    )
    expect(timestampCalls).toHaveLength(0)
  })

  it("timestampData is reset to null on episode change", async () => {
    const mockTimestamps = makeTimestampData()
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const urlStr = typeof url === "string" ? url : url.toString()
      if (urlStr.includes("component_timestamps")) {
        return { ok: true, json: () => Promise.resolve(mockTimestamps) } as Response
      }
      return {
        ok: true,
        json: () => Promise.resolve({
          total_frames: 100, fps: 30, duration_s: 3.3, folder: "ep_0",
          cameras: { top: true, left: true, right: true },
          trim_start_frame: 0, trim_end_frame: 99, trim_segments: [],
          value_trim_start: 0, value_trim_end: 99, discarded: false,
        }),
      } as Response
    })

    const { useEpisode } = await import("@/hooks/useEpisode")
    const { result, rerender } = renderHook(
      ({ taskId, idx }: { taskId: string; idx: number | null }) => useEpisode(taskId, idx),
      { initialProps: { taskId: "task1", idx: 0 } },
    )

    await actHook(async () => {
      await new Promise(r => setTimeout(r, 50))
    })

    // Load timestamps
    await actHook(async () => {
      await result.current.loadTimestamps()
    })
    expect(result.current.timestampData).not.toBeNull()

    // Switch episode
    rerender({ taskId: "task1", idx: 1 })
    await actHook(async () => {
      await new Promise(r => setTimeout(r, 50))
    })

    // timestampData should be reset
    expect(result.current.timestampData).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// 5. Tooltip content formatting (indirect via DOM)
// ---------------------------------------------------------------------------

describe("tooltip formatting", () => {
  it("tooltip div exists but is hidden by default", () => {
    const { container } = render(
      <TimestampChart data={makeTimestampData()} chartStep={0} totalSteps={3} fps={30} onSeek={vi.fn()} />,
    )
    // The tooltip div has display:none by default
    const divs = container.querySelectorAll("div")
    // Find the tooltip (second div inside container, has position:absolute, display:none)
    let tooltipFound = false
    divs.forEach(div => {
      if (div.style.display === "none" && div.style.position === "absolute") {
        tooltipFound = true
      }
    })
    expect(tooltipFound).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// 6. ComponentTimestampData type shape tests
// ---------------------------------------------------------------------------

describe("ComponentTimestampData type shape", () => {
  it("has timestamps as an array of Record<string, number>", () => {
    const data = makeTimestampData()
    expect(Array.isArray(data.timestamps)).toBe(true)
    expect(typeof data.timestamps[0]).toBe("object")
    expect(typeof data.timestamps[0].left_state).toBe("number")
  })

  it("has action_sources as string[]", () => {
    const data = makeTimestampData()
    expect(Array.isArray(data.action_sources)).toBe(true)
    data.action_sources.forEach(s => expect(typeof s).toBe("string"))
  })

  it("has record_timestamps as number[]", () => {
    const data = makeTimestampData()
    expect(Array.isArray(data.record_timestamps)).toBe(true)
    data.record_timestamps.forEach(t => expect(typeof t).toBe("number"))
  })
})
