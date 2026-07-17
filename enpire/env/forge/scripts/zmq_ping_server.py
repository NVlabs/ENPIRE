#!/usr/bin/env python3
import argparse

import zmq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"zmq ping server listening on tcp://{args.host}:{args.port}", flush=True)

    while True:
        msg = sock.recv()
        print(f"received: {msg!r}", flush=True)
        sock.send(b"pong")


if __name__ == "__main__":
    main()
