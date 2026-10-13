#!/usr/bin/env python3
"""Merge benchmark-local IRIS error records into the append-only corpus."""

import argparse
import fcntl
import json
import os
from pathlib import Path


def records(path):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("bench") or "attempt" not in record:
                raise ValueError("missing bench or attempt")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        yield json.dumps(record, sort_keys=True, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    state = Path(os.environ.get("PORTING_STATE_DIR",
                               os.environ.get("STATE", str(repo / ".porting-state"))))
    target = state / "iris" / "error_corpus.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / "error_corpus.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        known = set(records(target)) if target.exists() else set()
        new = []
        sources = sorted((state / "porting_logs").glob("*-iris/error_corpus.jsonl"))
        for source in sources:
            for row in records(source):
                if row not in known:
                    new.append(row)
                    known.add(row)
        if new and not args.dry_run:
            # Each whole JSON record is appended; old research records are retained.
            needs_newline = False
            if target.exists() and target.stat().st_size:
                with target.open("rb") as stream:
                    stream.seek(-1, 2)
                    needs_newline = stream.read(1) != b"\n"
            with target.open("a") as stream:
                if needs_newline:
                    stream.write("\n")
                for row in new:
                    stream.write(row + "\n")
        print(f"{'Pending' if args.dry_run else 'Appended'} {len(new)} records "
              f"from {len(sources)} local corpora; unique total {len(known)}.")


if __name__ == "__main__":
    main()
