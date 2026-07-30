// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CheckCircle2, FolderOpen } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Props {
  folderPath: string | null
  annotatedCount: number
  discardedCount: number
  totalCount: number
  onOpenFolder: () => void
  onReturn?: () => void
}

export function FinishedAnnotation({
  folderPath,
  annotatedCount,
  discardedCount,
  totalCount,
  onOpenFolder,
  onReturn,
}: Props) {
  return (
    <div className="mb-6 bg-card border rounded-xl p-10 text-center">
      <CheckCircle2 className="h-16 w-16 mx-auto mb-4 text-green-500" />
      <h2 className="text-2xl font-semibold mb-2">Finished annotation</h2>
      <div className="flex items-center justify-center gap-4 text-sm font-mono mb-6">
        <span className="text-green-500">✓ {annotatedCount} annotated</span>
        <span className="opacity-30">·</span>
        <span className="text-red-500">✕ {discardedCount} discarded</span>
        <span className="opacity-30">·</span>
        <span className="text-muted-foreground">{totalCount} total</span>
      </div>
      <div className="flex flex-col items-center gap-3">
        {folderPath && (
          <code className="text-xs bg-muted px-3 py-2 rounded font-mono break-all max-w-full text-muted-foreground">
            {folderPath}
          </code>
        )}
        <Button
          variant="outline"
          onClick={onOpenFolder}
          disabled={!folderPath}
          className="gap-2"
        >
          <FolderOpen className="h-4 w-4" /> Open folder
        </Button>
        {onReturn && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onReturn}
            className="text-xs text-muted-foreground"
          >
            Back to last episode
          </Button>
        )}
      </div>
    </div>
  )
}
