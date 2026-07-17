import { useRef, type ReactNode, type DragEvent } from "react"
import { GripVertical } from "lucide-react"

interface Props {
  panelId: string
  children: ReactNode
  onDragStart: (id: string) => void
  onDragOver: (id: string) => void
  onDragEnd: () => void
  isDragTarget: boolean
  dragPosition: "before" | "after" | null
}

export function DraggablePanel({ panelId, children, onDragStart, onDragOver, onDragEnd, isDragTarget, dragPosition }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  const handleDragStart = (e: DragEvent) => {
    e.dataTransfer.effectAllowed = "move"
    e.dataTransfer.setData("text/plain", panelId)
    onDragStart(panelId)
  }

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
    onDragOver(panelId)
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
  }

  const borderClass = isDragTarget
    ? dragPosition === "before"
      ? "border-t-2 border-t-blue-500"
      : "border-b-2 border-b-blue-500"
    : ""

  return (
    <div
      ref={ref}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={`relative group ${borderClass}`}
    >
      <div
        draggable
        onDragStart={handleDragStart}
        onDragEnd={onDragEnd}
        className="absolute left-0 top-0 bottom-0 w-6 flex items-center justify-center cursor-grab active:cursor-grabbing opacity-30 hover:opacity-100 z-10 transition-opacity"
        title="Drag to reorder"
      >
        <GripVertical className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="pl-2">
        {children}
      </div>
    </div>
  )
}
