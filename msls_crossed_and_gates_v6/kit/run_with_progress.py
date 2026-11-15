#!/usr/bin/env python3
"""Run a command with streamed output, persistent logging, and heartbeats."""

from __future__ import annotations

import argparse
import selectors
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence


def run_with_progress(
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    heartbeat_seconds: float = 30.0,
) -> int:
    if not command:
        raise ValueError("command must not be empty")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")

    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    printable = shlex.join([str(part) for part in command])
    start = time.monotonic()
    last_output = start

    print("=" * 100)
    print("$", printable)
    print("Live log:", log_path)
    print("=" * 100, flush=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"COMMAND:\n{printable}\n\nOUTPUT:\n")

        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            process.kill()
            raise RuntimeError("Failed to capture subprocess output")

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        try:
            while True:
                events = selector.select(timeout=1.0)

                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="", flush=True)
                        log_file.write(line)
                        last_output = time.monotonic()

                now = time.monotonic()
                if process.poll() is None and now - last_output >= heartbeat_seconds:
                    elapsed = (now - start) / 60.0
                    heartbeat = (
                        f"[progress] Process still running — "
                        f"elapsed {elapsed:.1f} minutes\n"
                    )
                    print(heartbeat, end="", flush=True)
                    log_file.write(heartbeat)
                    last_output = now

                if process.poll() is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        print(remaining, end="", flush=True)
                        log_file.write(remaining)
                    break
        finally:
            selector.unregister(process.stdout)
            process.stdout.close()

        return_code = process.wait()
        elapsed = (time.monotonic() - start) / 60.0
        completion = (
            "\n" + "=" * 100 + "\n"
            f"Return code: {return_code}\n"
            f"Elapsed time: {elapsed:.2f} minutes\n"
            f"Log saved to: {log_path}\n"
            + "=" * 100 + "\n"
        )
        print(completion, flush=True)
        log_file.write(completion)

    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("A command is required after --")
    return args


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run_with_progress(
            args.command,
            args.log,
            cwd=args.cwd,
            heartbeat_seconds=args.heartbeat_seconds,
        )
    )


if __name__ == "__main__":
    main()
