import argparse
import gzip
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTMSS = ROOT.parent / "FastMSS"


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_rttm(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            rows.append(fields)
    return rows


def build_records(manifest_dir, sim_root):
    records = {}
    for split in ("train", "dev"):
        manifest = manifest_dir / f"synth-khmer-{split}.jsonl.gz"
        if not manifest.exists():
            raise SystemExit(f"Missing manifest: {manifest}")
        for cut in read_jsonl_gz(manifest):
            rec_id = cut["recording"]["id"]
            if rec_id in records:
                continue
            wav = Path(cut["recording"]["sources"][0]["source"])
            duration = float(cut["duration"])
            supervisions = cut["supervisions"]
            speakers = tuple(sorted({item["speaker"] for item in supervisions}))
            source_cuts = tuple(
                sorted(
                    {
                        item.get("custom", {}).get("original_cut_id", "")
                        for item in supervisions
                        if item.get("custom", {}).get("original_cut_id")
                    }
                )
            )
            records[rec_id] = {
                "rec_id": rec_id,
                "wav": wav,
                "duration": duration,
                "speakers": speakers,
                "source_cuts": source_cuts,
            }

    rttm_map = {path.stem: path for path in sim_root.glob("*/rttm/*.rttm")}
    missing_rttm = sorted(rec_id for rec_id in records if rec_id not in rttm_map)
    if missing_rttm:
        raise SystemExit(f"Missing RTTM for {len(missing_rttm)} recordings, first: {missing_rttm[0]}")
    return records, rttm_map


def validate_records(records, rttm_map):
    errors = []
    for rec_id, info in records.items():
        if not info["wav"].exists():
            errors.append(f"{rec_id}: missing WAV {info['wav']}")
            continue
        if " " in str(info["wav"]):
            errors.append(f"{rec_id}: WAV path contains spaces and will break wav.scp")
        rows = parse_rttm(rttm_map[rec_id])
        if not rows:
            errors.append(f"{rec_id}: empty RTTM")
            continue
        rttm_speakers = {row[7] for row in rows if len(row) >= 8}
        if rttm_speakers != set(info["speakers"]):
            errors.append(f"{rec_id}: RTTM speakers {sorted(rttm_speakers)} != manifest {sorted(info['speakers'])}")
        for row in rows:
            if row[1] != rec_id:
                errors.append(f"{rec_id}: RTTM session id {row[1]} does not match")
            start = float(row[3])
            duration = float(row[4])
            if start < 0 or duration <= 0:
                errors.append(f"{rec_id}: invalid RTTM interval {start} {duration}")
            if start + duration > info["duration"] + 0.1:
                errors.append(f"{rec_id}: RTTM interval exceeds duration")
    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"Validation failed with {len(errors)} errors:\n{preview}")


def assign_holdout(records, test_speakers, dev_speakers, seed):
    all_speakers = sorted({speaker for info in records.values() for speaker in info["speakers"]})
    if test_speakers + dev_speakers >= len(all_speakers):
        raise SystemExit("Not enough speakers for the requested holdout")
    rng = random.Random(seed)
    rng.shuffle(all_speakers)
    test_set = set(all_speakers[:test_speakers])
    dev_set = set(all_speakers[test_speakers : test_speakers + dev_speakers])
    train_set = set(all_speakers[test_speakers + dev_speakers :])
    splits = {"train": [], "dev": [], "test": []}
    for rec_id, info in records.items():
        meeting_speakers = set(info["speakers"])
        if meeting_speakers & test_set:
            splits["test"].append(rec_id)
        elif meeting_speakers & dev_set:
            splits["dev"].append(rec_id)
        else:
            splits["train"].append(rec_id)
    return splits, {"train": train_set, "dev": dev_set, "test": test_set}


def assign_folds(records, folds, seed):
    all_speakers = sorted({speaker for info in records.values() for speaker in info["speakers"]})
    if folds < 2 or folds > len(all_speakers):
        raise SystemExit("Invalid fold count")
    rng = random.Random(seed)
    rng.shuffle(all_speakers)
    speaker_folds = [set(all_speakers[index::folds]) for index in range(folds)]
    split_folds = {}
    for index in range(folds):
        test_set = speaker_folds[index]
        dev_set = speaker_folds[(index + 1) % folds]
        splits = {"train": [], "dev": [], "test": []}
        for rec_id, info in records.items():
            meeting_speakers = set(info["speakers"])
            if meeting_speakers & test_set:
                splits["test"].append(rec_id)
            elif meeting_speakers & dev_set:
                splits["dev"].append(rec_id)
            else:
                splits["train"].append(rec_id)
        split_folds[index] = {
            "splits": splits,
            "speakers": {"train": set(all_speakers) - test_set - dev_set, "dev": dev_set, "test": test_set},
        }
    return split_folds


def limit_train(records, rec_ids, max_hours, seed):
    if max_hours is None or max_hours <= 0:
        return list(rec_ids)
    rng = random.Random(seed)
    shuffled = list(rec_ids)
    rng.shuffle(shuffled)
    all_speakers = {speaker for rec_id in rec_ids for speaker in records[rec_id]["speakers"]}
    selected = []
    selected_set = set()
    speaker_seen = set()
    for rec_id in shuffled:
        new_speakers = set(records[rec_id]["speakers"]) - speaker_seen
        if new_speakers:
            selected.append(rec_id)
            selected_set.add(rec_id)
            speaker_seen.update(records[rec_id]["speakers"])
            if speaker_seen == all_speakers:
                break
    for rec_id in shuffled:
        if rec_id in selected_set:
            continue
        selected.append(rec_id)
        selected_set.add(rec_id)
        hours = sum(records[item]["duration"] for item in selected) / 3600.0
        if hours >= max_hours:
            break
    return selected


def split_stats(records, rec_ids):
    speakers = {speaker for rec_id in rec_ids for speaker in records[rec_id]["speakers"]}
    source_cuts = {cut for rec_id in rec_ids for cut in records[rec_id]["source_cuts"]}
    hours = sum(records[rec_id]["duration"] for rec_id in rec_ids) / 3600.0
    return {
        "meetings": len(rec_ids),
        "hours": round(hours, 3),
        "speakers": sorted(speakers),
        "num_speakers": len(speakers),
        "source_cuts": len(source_cuts),
    }


def write_split(out_dir, records, rttm_map, rec_ids):
    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir = out_dir / "rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "wav.scp"
    rttm_all_path = out_dir / "rttm.txt"
    uem_path = out_dir / "all.uem"
    with wav_path.open("w", encoding="utf-8") as wav_handle, rttm_all_path.open(
        "w", encoding="utf-8"
    ) as rttm_all_handle, uem_path.open("w", encoding="utf-8") as uem_handle:
        for rec_id in sorted(rec_ids):
            info = records[rec_id]
            wav_handle.write(f"{rec_id} {info['wav'].as_posix()}\n")
            uem_handle.write(f"{rec_id} 1 0.000 {info['duration']:.6f}\n")
            lines = [
                line.strip() + "\n"
                for line in rttm_map[rec_id].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            (rttm_dir / f"{rec_id}.rttm").write_text("".join(lines), encoding="utf-8")
            rttm_all_handle.writelines(lines)


def leakage_report(records, splits, designated):
    train_ids = set(splits["train"])
    dev_ids = set(splits["dev"])
    test_ids = set(splits["test"])
    train_cuts = {cut for rec_id in train_ids for cut in records[rec_id]["source_cuts"]}
    dev_cuts = {cut for rec_id in dev_ids for cut in records[rec_id]["source_cuts"]}
    test_cuts = {cut for rec_id in test_ids for cut in records[rec_id]["source_cuts"]}
    test_in_train = [
        rec_id for rec_id in train_ids if set(records[rec_id]["speakers"]) & designated["test"]
    ]
    dev_in_train = [
        rec_id for rec_id in train_ids if set(records[rec_id]["speakers"]) & designated["dev"]
    ]
    return {
        "test_speakers_in_train_meetings": len(test_in_train),
        "dev_speakers_in_train_meetings": len(dev_in_train),
        "source_cut_overlap_train_dev": len(train_cuts & dev_cuts),
        "source_cut_overlap_train_test": len(train_cuts & test_cuts),
    }


def write_bundle(out_root, records, rttm_map, splits, designated, metadata):
    bundle = out_root
    stats = {}
    for split_name, rec_ids in splits.items():
        write_split(bundle / split_name, records, rttm_map, rec_ids)
        stats[split_name] = split_stats(records, rec_ids)
    report = {
        "mode": metadata["mode"],
        "seed": metadata["seed"],
        "folds": metadata.get("folds"),
        "max_train_hours": metadata.get("max_train_hours"),
        "splits": stats,
        "designated_speakers": {key: sorted(value) for key, value in designated.items()},
        "leakage": leakage_report(records, splits, designated),
    }
    (bundle / "split.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default=str(FASTMSS / "data" / "sim" / "manifests"))
    parser.add_argument("--sim-root", default=str(FASTMSS / "data" / "sim"))
    parser.add_argument("--out", default=str(ROOT / "data" / "diarizen" / "khmer"))
    parser.add_argument("--mode", choices=("holdout", "cv"), default="holdout")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-speakers", type=int, default=3)
    parser.add_argument("--dev-speakers", type=int, default=3)
    parser.add_argument("--max-train-hours", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    sim_root = Path(args.sim_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    records, rttm_map = build_records(manifest_dir, sim_root)
    validate_records(records, rttm_map)
    print(f"Loaded {len(records)} meetings with {len({speaker for info in records.values() for speaker in info['speakers']})} speakers")

    if args.mode == "holdout":
        splits, designated = assign_holdout(records, args.test_speakers, args.dev_speakers, args.seed)
        splits["train"] = limit_train(records, splits["train"], args.max_train_hours, args.seed)
        report = write_bundle(
            out_root / "holdout",
            records,
            rttm_map,
            splits,
            designated,
            {
                "mode": args.mode,
                "seed": args.seed,
                "max_train_hours": args.max_train_hours,
            },
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        split_folds = assign_folds(records, args.folds, args.seed)
        for index, payload in split_folds.items():
            splits = payload["splits"]
            splits["train"] = limit_train(records, splits["train"], args.max_train_hours, args.seed)
            report = write_bundle(
                out_root / f"fold_{index}",
                records,
                rttm_map,
                splits,
                payload["speakers"],
                {
                    "mode": args.mode,
                    "seed": args.seed,
                    "folds": args.folds,
                    "max_train_hours": args.max_train_hours,
                },
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
