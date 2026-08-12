/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

import { useCallback, useEffect, useRef, useState } from "react";

export type WSMessageType =
  | "state_update"
  | "code_proposal"
  | "execution_log"
  | "camera_frame"
  | "error"
  | "status_change"
  | "detection_debug"
  | "segmentation_debug"
  | "vlm_result"
  | "policy_update"
  | "debug_mode_change"
  | "learn_skill_update"
  | "grasp_viz"
  | "motion_planner_debug"
  | "contact_detect"
  | "stdout_stream";

export interface WSMessage {
  type: WSMessageType;
  data: unknown;
  timestamp: string;
}

interface UseWebSocketOptions {
  url?: string;
  reconnectInterval?: number;
  maxRetries?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`,
    reconnectInterval = 2000,
    maxRetries = 10,
  } = options;

  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const listenersRef = useRef<Map<WSMessageType, Set<(data: unknown) => void>>>(
    new Map(),
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      retriesRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as WSMessage;
        setLastMessage(msg);
        const listeners = listenersRef.current.get(msg.type);
        if (listeners) {
          listeners.forEach((cb) => cb(msg.data));
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      if (retriesRef.current < maxRetries) {
        retriesRef.current += 1;
        setTimeout(connect, reconnectInterval);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [url, reconnectInterval, maxRetries]);

  const subscribe = useCallback(
    (type: WSMessageType, callback: (data: unknown) => void) => {
      if (!listenersRef.current.has(type)) {
        listenersRef.current.set(type, new Set());
      }
      listenersRef.current.get(type)!.add(callback);
      return () => {
        listenersRef.current.get(type)?.delete(callback);
      };
    },
    [],
  );

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected, lastMessage, subscribe, send };
}
