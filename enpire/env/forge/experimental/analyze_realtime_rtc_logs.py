from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def _event_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        event = str(record.get("event"))
        counts[event] = counts.get(event, 0) + 1
    return counts


def _first_scalar(action: dict[str, Any], key: str = "left_joint_pos") -> float | None:
    value = action.get(key)
    if not isinstance(value, list) or not value:
        return None
    item = value[0]
    if isinstance(item, (int, float)):
        return float(item)
    return None


def _prefix_series(record: dict[str, Any], key: str = "left_joint_pos") -> list[float]:
    rtc_payload = record.get("rtc_payload") or {}
    prefix_chunk = rtc_payload.get("prefix_action_chunk") or {}
    values = prefix_chunk.get(key)
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for row in values:
        if isinstance(row, list) and row:
            result.append(float(row[0]))
    return result


def analyze_client(records: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    by_event: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_event.setdefault(str(record.get("event")), []).append(record)

    scheduled = by_event.get("request_scheduled", [])
    activated = {int(r["request_id"]): r for r in by_event.get("request_activated", []) if r.get("request_id") is not None}
    completed = {int(r["request_id"]): r for r in by_event.get("request_completed", []) if r.get("request_id") is not None}
    step_events = by_event.get("step_executed", [])
    max_request_id = max(int(r["request_id"]) for r in scheduled)

    if not scheduled:
        messages.append("FAIL: no client request_scheduled events found")
        return messages

    bootstrap = scheduled[0]
    if int(bootstrap.get("d_sent", -1)) != 0:
        messages.append(f"FAIL: first scheduled request should be bootstrap with d_sent=0, got {bootstrap.get('d_sent')}")
    else:
        messages.append("PASS: bootstrap request logged with d_sent=0")

    for record in scheduled:
        request_id = int(record["request_id"])
        d_sent = int(record.get("d_sent", 0))
        reason = str(record.get("reason"))
        activation = activated.get(request_id)
        completion = completed.get(request_id)
        if completion is None:
            level = "WARN" if request_id == max_request_id else "FAIL"
            messages.append(f"{level}: request {request_id} ({reason}) never completed")
            continue
        if activation is None:
            level = "WARN" if request_id == max_request_id else "FAIL"
            messages.append(f"{level}: request {request_id} ({reason}) never activated")
            continue

        start_exec_index = int(activation.get("start_exec_index", -1))
        if start_exec_index != d_sent:
            messages.append(
                f"FAIL: request {request_id} activated at chunk index {start_exec_index}, expected {d_sent}"
            )
        else:
            messages.append(f"PASS: request {request_id} activated at expected chunk index {d_sent}")

        if reason != "replan":
            continue

        old_request_id = record.get("active_request_id")
        if old_request_id is None:
            messages.append(f"FAIL: replan request {request_id} missing active_request_id")
            continue
        old_request_id = int(old_request_id)

        schedule_line = int(record["_line_no"])
        activation_line = int(activation["_line_no"])
        old_steps = [
            step for step in step_events
            if schedule_line < int(step["_line_no"]) < activation_line and int(step.get("active_request_id", -1)) == old_request_id
        ]
        if len(old_steps) < d_sent:
            messages.append(
                f"FAIL: request {request_id} expected at least {d_sent} old-chunk overlap steps, saw {len(old_steps)}"
            )
            continue

        expected_prefix = _prefix_series(record)
        observed_prefix = [
            _first_scalar(step.get("action", {}))
            for step in old_steps[:d_sent]
        ]
        if expected_prefix and observed_prefix != expected_prefix[:d_sent]:
            messages.append(
                f"FAIL: request {request_id} overlap mismatch; expected {expected_prefix[:d_sent]}, observed {observed_prefix}"
            )
        else:
            messages.append(
                f"PASS: request {request_id} overlap prefix matched for {d_sent} step(s)"
            )

        next_step = None
        for step in step_events:
            if int(step["_line_no"]) > activation_line and int(step.get("active_request_id", -1)) == request_id:
                next_step = step
                break
        if next_step is None:
            messages.append(f"FAIL: request {request_id} activated but no executed step followed")
        elif int(next_step.get("chunk_local_index", -1)) != d_sent:
            messages.append(
                f"FAIL: request {request_id} first executed new-chunk step index was {next_step.get('chunk_local_index')}, expected {d_sent}"
            )
        else:
            messages.append(
                f"PASS: request {request_id} started executing new chunk from index {d_sent}"
            )

    return messages


def analyze_server(client_records: list[dict[str, Any]], server_records: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    client_scheduled = {
        int(r["request_id"]): r
        for r in client_records
        if r.get("event") == "request_scheduled"
    }
    server_completed = {}
    for record in server_records:
        if record.get("event") != "request_completed":
            continue
        client_rtc = record.get("client_rtc") or {}
        request_id = client_rtc.get("request_id")
        if request_id is None:
            continue
        server_completed[int(request_id)] = record

    if not server_completed:
        messages.append("WARN: no server request_completed events with client request_id found")
        return messages

    for request_id, client_record in sorted(client_scheduled.items()):
        if int(client_record.get("d_sent", 0)) <= 0:
            continue
        server_record = server_completed.get(request_id)
        if server_record is None:
            messages.append(f"FAIL: no server completion record found for client request {request_id}")
            continue
        used_prefix_len = int(server_record.get("used_prefix_len", -1))
        d_sent = int(client_record.get("d_sent", 0))
        if used_prefix_len != d_sent:
            messages.append(
                f"FAIL: server used prefix_len={used_prefix_len} for request {request_id}, expected client d_sent={d_sent}"
            )
        else:
            messages.append(
                f"PASS: server used expected prefix_len={d_sent} for request {request_id}"
            )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-log", required=True, help="Client-side realtime RTC JSONL log")
    parser.add_argument("--server-log", default="", help="Optional server-side RTC JSONL log")
    args = parser.parse_args()

    client_records = _load_jsonl(args.client_log)
    print(f"Loaded {len(client_records)} client log records from {args.client_log}")
    print("Client event counts:", _event_counts(client_records))
    for message in analyze_client(client_records):
        print(message)

    if args.server_log:
        server_records = _load_jsonl(args.server_log)
        print(f"\nLoaded {len(server_records)} server log records from {args.server_log}")
        print("Server event counts:", _event_counts(server_records))
        for message in analyze_server(client_records, server_records):
            print(message)


if __name__ == "__main__":
    main()
