"""Build a Hugging Face dataset staging folder from the Khmer simulation.

The source audio remains in ``FastMSS/data/sim``.  This script validates the
DiariZen speaker-disjoint split, copies RTTMs, links or copies audio, and writes
portable JSONL metadata with relative ``audio`` and ``rttm`` paths.
"""

import argparse
import gzip
import json
import os
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FASTMSS = ROOT.parent / "FastMSS"
DIARIZEN_DATA = ROOT / "data" / "diarizen" / "khmer_full" / "holdout"
DEFAULT_OUT = ROOT.parent / "hf_khmer_diarization"
SPLITS = ("train", "dev", "test")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_manifests(fastmss):
    records = {}
    for name in ("synth-khmer-train.jsonl.gz", "synth-khmer-dev.jsonl.gz"):
        path = fastmss / "data" / "sim" / "manifests" / name
        if not path.exists():
            raise SystemExit(f"Missing manifest: {path}")
        for record in read_jsonl_gz(path):
            recording_id = record["recording"]["id"]
            if recording_id in records:
                raise SystemExit(f"Duplicate recording id: {recording_id}")
            records[recording_id] = record
    return records


def read_scp(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        recording_id, wav = line.split(maxsplit=1)
        rows[recording_id] = Path(wav)
    return rows


def parse_rttm(path):
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7] if parts[7] != "<NA>" else parts[-2]
        segments.append(
            {"start": start, "end": start + duration, "speaker": speaker}
        )
    return segments


def overlap_stats(segments):
    events = []
    for segment in segments:
        events.append((segment["start"], 1))
        events.append((segment["end"], -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    previous = None
    union = 0.0
    overlap = 0.0
    for time, delta in events:
        if previous is not None and time > previous:
            if active >= 1:
                union += time - previous
            if active >= 2:
                overlap += time - previous
        active += delta
        previous = time
    return union, overlap, (overlap / union if union else 0.0)


def segment_records(record):
    segments = []
    for supervision in record.get("supervisions") or []:
        custom = supervision.get("custom") or {}
        start = float(supervision["start"])
        duration = float(supervision["duration"])
        segments.append(
            {
                "start": round(start, 4),
                "end": round(start + duration, 4),
                "speaker": supervision["speaker"],
                "text": supervision.get("text", ""),
                "transition_type": custom.get("transition_type"),
                "source_cut_id": custom.get("original_cut_id"),
                "speech_level_db": custom.get("speech_level_db"),
            }
        )
    return segments


def summarize(records):
    if not records:
        return {}
    durations = [item["duration"] for item in records]
    overlap_ratios = [item["overlap_ratio"] for item in records]
    overlap_fractions = [item["overlap_fraction"] for item in records]
    speaker_counts = [item["num_speakers"] for item in records]
    speakers = sorted({speaker for item in records for speaker in item["speakers"]})
    source_cuts = {
        segment["source_cut_id"]
        for item in records
        for segment in item["segments"]
        if segment["source_cut_id"]
    }
    return {
        "meetings": len(records),
        "hours": round(sum(durations) / 3600.0, 3),
        "speakers": len(speakers),
        "source_cuts": len(source_cuts),
        "segments": sum(item["num_segments"] for item in records),
        "rttm_segments": sum(item["num_rttm_segments"] for item in records),
        "speaker_counts": {
            str(count): speaker_counts.count(count) for count in sorted(set(speaker_counts))
        },
        "duration_s": {
            "min": round(min(durations), 3),
            "mean": round(statistics.mean(durations), 3),
            "max": round(max(durations), 3),
        },
        "overlap_ratio": {
            "mean": round(statistics.mean(overlap_ratios), 4),
            "p95": round(sorted(overlap_ratios)[int(0.95 * (len(overlap_ratios) - 1))], 4),
        },
        "overlap_fraction": {
            "mean": round(statistics.mean(overlap_fractions), 4),
            "p95": round(
                sorted(overlap_fractions)[int(0.95 * (len(overlap_fractions) - 1))], 4
            ),
        },
    }


def copy_audio(source, target, mode):
    if mode == "none" or target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastmss", type=Path, default=FASTMSS)
    parser.add_argument("--diarizen-data", type=Path, default=DIARIZEN_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--audio-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="Hardlink audio, copy it, or build metadata/RTTM only.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit meetings per split.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    fastmss = args.fastmss.expanduser().resolve()
    diarizen_data = args.diarizen_data.expanduser().resolve()
    out = args.out.expanduser().resolve()
    metadata_dir = out / "metadata"
    rttm_dir = out / "rttm"
    audio_dir = out / "audio"

    manifests = load_manifests(fastmss)
    all_stats = {}
    all_records = {}
    errors = []
    warnings = []

    if not args.dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if args.audio_mode != "none":
            audio_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_root = diarizen_data / split
        scp_path = split_root / "wav.scp"
        if not scp_path.exists():
            raise SystemExit(f"Missing wav.scp: {scp_path}")
        rows = read_scp(scp_path)
        records = []
        if not args.dry_run:
            (rttm_dir / split).mkdir(parents=True, exist_ok=True)

        for recording_id, wav_path in rows.items():
            if args.limit and len(records) >= args.limit:
                break
            if recording_id not in manifests:
                errors.append(f"{split}/{recording_id}: missing synth manifest")
                continue
            if not wav_path.exists():
                errors.append(f"{split}/{recording_id}: missing audio {wav_path}")
                continue
            source_rttm = split_root / "rttm" / f"{recording_id}.rttm"
            if not source_rttm.exists():
                errors.append(f"{split}/{recording_id}: missing RTTM {source_rttm}")
                continue

            info = sf.info(str(wav_path))
            if info.samplerate != 16000:
                errors.append(f"{split}/{recording_id}: sample rate {info.samplerate}")
            if info.channels != 1:
                errors.append(f"{split}/{recording_id}: channels {info.channels}")

            manifest = manifests[recording_id]
            segments = segment_records(manifest)
            rttm_segments = parse_rttm(source_rttm)
            manifest_speakers = {segment["speaker"] for segment in segments}
            rttm_speakers = {segment["speaker"] for segment in rttm_segments}
            if not segments:
                errors.append(f"{split}/{recording_id}: no supervisions")
            if not rttm_segments:
                errors.append(f"{split}/{recording_id}: empty RTTM")
            if rttm_speakers != manifest_speakers:
                errors.append(
                    f"{split}/{recording_id}: speaker mismatch "
                    f"{sorted(rttm_speakers)} != {sorted(manifest_speakers)}"
                )

            audio_duration = info.frames / info.samplerate
            manifest_duration = float(manifest["recording"]["duration"])
            if abs(audio_duration - manifest_duration) > 0.1:
                warnings.append(
                    f"{split}/{recording_id}: audio {audio_duration:.3f}s "
                    f"!= manifest {manifest_duration:.3f}s"
                )

            for segment in rttm_segments:
                if (
                    segment["start"] < 0
                    or segment["end"] <= segment["start"]
                    or segment["end"] > audio_duration + 0.1
                ):
                    errors.append(
                        f"{split}/{recording_id}: invalid RTTM interval "
                        f"{segment['start']} {segment['end']} {segment['speaker']}"
                    )

            union, overlap, ratio = overlap_stats(segments)
            overlap_fraction = overlap / audio_duration if audio_duration else 0.0
            topic = wav_path.parent.parent.name
            record = {
                "meeting_id": recording_id,
                "split": split,
                "audio": f"audio/{recording_id}.wav",
                "rttm": f"rttm/{split}/{recording_id}.rttm",
                "duration": round(audio_duration, 4),
                "sampling_rate": info.samplerate,
                "channels": info.channels,
                "topic": topic,
                "num_speakers": len(manifest_speakers),
                "speakers": sorted(manifest_speakers),
                "num_segments": len(segments),
                "num_rttm_segments": len(rttm_segments),
                "speech_seconds": round(union, 4),
                "overlap_seconds": round(overlap, 4),
                "overlap_ratio": round(ratio, 4),
                "overlap_fraction": round(overlap_fraction, 4),
                "segments": segments,
            }
            records.append(record)

            if not args.dry_run:
                shutil.copy2(source_rttm, rttm_dir / split / f"{recording_id}.rttm")
                copy_audio(wav_path, audio_dir / f"{recording_id}.wav", args.audio_mode)

        all_records[split] = records
        all_stats[split] = summarize(records)
        if not args.dry_run:
            write_jsonl(metadata_dir / f"{split}.jsonl", records)

    combined = [item for split in SPLITS for item in all_records[split]]
    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(fastmss / "data" / "sim"),
        "source_corpus": "Digital-Divide-Data/khmer-speech-dataset (CC BY-SA 4.0)",
        "diarizen_data": str(diarizen_data),
        "audio_mode": args.audio_mode,
        "splits": all_stats,
        "total": summarize(combined),
        "errors": errors,
        "warnings": warnings,
        "notes": [
            "segments are utterance-level supervisions from the FastMSS manifests.",
            "RTTM is the reference diarization, derived from the simulation's "
            "aligned speech segments and merged with a 0.2 s same-speaker gap "
            "threshold; RTTM segment counts therefore differ from manifest "
            "segment counts.",
            "overlap_ratio = overlap_seconds / union_speech_seconds; "
            "overlap_fraction = overlap_seconds / meeting_duration.",
        ],
    }

    if not args.dry_run:
        write_json(metadata_dir / "stats.json", stats)
        write_json(metadata_dir / "split.json", json.loads((diarizen_data / "split.json").read_text(encoding="utf-8")))
        validation = fastmss / "data" / "sim" / "manifests" / "validation.json"
        if validation.exists():
            shutil.copy2(validation, metadata_dir / "source_validation.json")
        print(f"Wrote {out}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if errors:
        print(f"{len(errors)} errors", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
