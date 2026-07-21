import "@testing-library/jest-dom/vitest"

// Stub canvas getContext for jsdom (no real Canvas in jsdom)
HTMLCanvasElement.prototype.getContext = (() => {
  const noop = () => {}
  const noopReturn = () => null
  return function (this: HTMLCanvasElement) {
    return {
      clearRect: noop,
      fillRect: noop,
      fillText: noop,
      beginPath: noop,
      moveTo: noop,
      lineTo: noop,
      stroke: noop,
      fill: noop,
      arc: noop,
      setTransform: noop,
      setLineDash: noop,
      save: noop,
      restore: noop,
      measureText: () => ({ width: 0 }),
      getImageData: noopReturn,
      putImageData: noop,
      canvas: this,
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 1,
      font: "",
      textAlign: "start" as CanvasTextAlign,
      textBaseline: "alphabetic" as CanvasTextBaseline,
      globalAlpha: 1,
    } as unknown as CanvasRenderingContext2D
  }
})()

// Stub ResizeObserver
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver

// Stub getBoundingClientRect for canvas elements
if (!Element.prototype.getBoundingClientRect) {
  Element.prototype.getBoundingClientRect = () => ({
    x: 0, y: 0, width: 800, height: 320, top: 0, right: 800, bottom: 320, left: 0,
    toJSON() { return this },
  })
}
