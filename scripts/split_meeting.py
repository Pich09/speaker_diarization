"""Split a real single-mic meeting into approximate per-speaker WAVs.

The input RTTM comes from diarization (DiariZen or another system).  For each
speaker this script:

1. picks a clean enrollment interval from target-only diarization regions,
2. calls ``extract_target.py`` with that enrollment and the RTTM,
3. writes ``<out-dir>/<speaker>.wav`` and a manifest.

This is model-based separation, not exact source recovery.  Overlap, more than
three active speakers, noise, and domain mismatch reduce quality.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTOR = SCRIPT_DIR / "extract_target.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--rttm", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--enroll-dir",
        type=Path,
        help="Optional directory of clean <speaker>.wav enrollment clips.",
    )
    parser.add_argument("--num-speakers", type=int, help="Defaults to unique RTTM speakers.")
    parser.add_argument("--min-enroll", type=float, default=0.5)
    parser.add_argument("--model")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-level-match", action="store_true")
    return parser.parse_args()


def safe_name(raw):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def parse_rttm(path):
    speakers = {}
    order = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        raw = parts[-2] if parts[-2] != "<NA>" else parts[-3]
        start = float(parts[3])
        end = start + float(parts[4])
        if raw not in speakers:
            speakers[raw] = []
            order.append(raw)
        speakers[raw].append((start, end))
    if not order:
        raise SystemExit(f"No SPEAKER lines in {path}")
    return order, speakers


def subtract(intervals, others):
    pieces = list(intervals)
    for other_start, other_end in others:
        next_pieces = []
        for start, end in pieces:
            if other_end <= start or other_start >= end:
                next_pieces.append((start, end))
                continue
            if start < other_start:
                next_pieces.append((start, other_start))
            if end > other_end:
                next_pieces.append((other_end, end))
        pieces = next_pieces
    return pieces


def pick_enrollment(target, others):
    exclusive = subtract(target, others)
    candidates = exclusive if exclusive else target
    start, end = max(candidates, key=lambda item: item[1] - item[0])
    return start, end, bool(exclusive)


def main():
    args = parse_args()
    order, speakers = parse_rttm(args.rttm)
    total = args.num_speakers or len(order)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enroll_dir = args.out_dir / "_enroll"
    enroll_dir.mkdir(parents=True, exist_ok=True)

    mix, sample_rate = sf.read(args.mix, dtype="float32", always_2d=True)
    if mix.shape[1] != 1:
        mix = mix.mean(axis=1, keepdims=True)
    mix = mix[:, 0]

    manifest = {
        "mix": str(args.mix),
        "rttm": str(args.rttm),
        "sample_rate": sample_rate,
        "num_speakers": total,
        "mode": "model-based target extraction",
        "speakers": [],
    }
    failures = 0

    for raw in order:
        target = speakers[raw]
        name = safe_name(raw)
        provided = args.enroll_dir / f"{name}.wav" if args.enroll_dir else None
        if provided and provided.exists():
            enroll_path = provided
            start = end = None
            exclusive = None
            enrollment_source = "provided"
        else:
            others = [
                item for other, values in speakers.items() if other != raw for item in values
            ]
            start, end, exclusive = pick_enrollment(target, others)
            if end - start < args.min_enroll:
                print(f"skip {raw}: enrollment interval {end - start:.2f}s is too short")
                manifest["speakers"].append(
                    {"raw": raw, "status": "skipped", "reason": "enrollment too short"}
                )
                failures += 1
                continue
            enroll_path = enroll_dir / f"{name}.wav"
            begin = int(start * sample_rate)
            finish = int(end * sample_rate)
            sf.write(enroll_path, mix[begin:finish], sample_rate, subtype="PCM_16")
            enrollment_source = "auto"
            if not exclusive:
                print(f"warning {raw}: enrollment overlaps other speakers")

        out_path = args.out_dir / f"{name}.wav"
        command = [
            sys.executable,
            str(EXTRACTOR),
            "--mix",
            str(args.mix),
            "--enroll",
            str(enroll_path),
            "--out",
            str(out_path),
            "--rttm",
            str(args.rttm),
            "--speaker",
            raw,
            "--num-speakers",
            str(total),
            "--threshold",
            str(args.threshold),
            "--chunk-seconds",
            str(args.chunk_seconds),
            "--overlap-seconds",
            str(args.overlap_seconds),
            "--passes",
            str(args.passes),
            "--device",
            args.device,
        ]
        if args.model:
            command.extend(["--model", args.model])
        if args.no_level_match:
            command.append("--no-level-match")

        if start is None:
            print(f"extract {raw} enrollment=provided")
        else:
            print(f"extract {raw} enrollment={start:.2f}-{end:.2f}s exclusive={exclusive}")
        result = subprocess.run(command, check=False)
        status = "ok" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            failures += 1
        manifest["speakers"].append(
            {
                "raw": raw,
                "status": status,
                "audio": str(out_path),
                "enrollment": str(enroll_path),
                "enrollment_source": enrollment_source,
                "enrollment_start": start,
                "enrollment_end": end,
                "exclusive_enrollment": exclusive,
                "target_intervals": len(target),
                "target_seconds": sum(item[1] - item[0] for item in target),
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    if failures:
        raise SystemExit(f"{failures} speaker(s) failed or were skipped")


if __name__ == "__main__":
    main()

