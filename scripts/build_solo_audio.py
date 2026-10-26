"""Rebuild exact per-speaker solo tracks for the synthetic dev demo.

The FastMSS simulator was run with ``reverberate: false``, ``use_fir: false``
and ``add_noise: false``.  Each meeting is therefore just a sum of source FLAC
cuts, placed at the offsets recorded in the synth manifest and RMS-normalised
to the recorded ``speech_level_db``.  This script replays that process and
writes one target-only WAV per speaker.
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "dev_demo"
INPUT_SCP = DEMO / "input_all.scp"
AUDIO_DIR = DEMO / "audio"
SOLO_DIR = DEMO / "solo"
OUT = DEMO / "solo_manifest.json"

FASTMSS = ROOT.parent / "FastMSS"
SIM_MANIFEST = FASTMSS / "data" / "sim" / "manifests" / "synth-khmer-train.jsonl.gz"
SOURCE_MANIFEST = FASTMSS / "data" / "manifests" / "khmer_all.jsonl.gz"


def load_meeting_ids():
    ids = []
    for line in INPUT_SCP.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(line.split()[0])
    return ids


def load_demo_records(meeting_ids):
    wanted = set(meeting_ids)
    records = {}
    with gzip.open(SIM_MANIFEST, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            supervisions = record.get("supervisions") or []
            if not supervisions:
                continue
            recording_id = supervisions[0].get("recording_id")
            if recording_id in wanted and recording_id not in records:
                records[recording_id] = record
    missing = wanted - set(records)
    if missing:
        raise RuntimeError(f"Missing synth records: {sorted(missing)}")
    return records


def needed_cut_prefixes(records):
    prefixes = set()
    for record in records.values():
        for supervision in record["supervisions"]:
            original_cut_id = supervision["custom"]["original_cut_id"]
            prefixes.add(original_cut_id.rsplit("-", 1)[0])
    return prefixes


def load_source_cuts(prefixes):
    sources = {}
    with gzip.open(SOURCE_MANIFEST, "rt", encoding="utf-8") as handle:
        for line in handle:
            cut = json.loads(line)
            cut_id = cut.get("id", "")
            prefix = cut_id.rsplit("-", 1)[0]
            if prefix not in prefixes:
                continue
            if prefix in sources:
                raise RuntimeError(f"Ambiguous source cut prefix: {prefix}")
            sources[prefix] = {
                "cut_id": cut_id,
                "source": cut["recording"]["sources"][0]["source"],
                "duration": float(cut["duration"]),
                "sampling_rate": int(cut["recording"]["sampling_rate"]),
            }
    missing = prefixes - set(sources)
    if missing:
        raise RuntimeError(f"Missing source cuts: {sorted(missing)}")
    return sources


def read_mono(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio[:, 0], sample_rate


def rms_db(audio):
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def safe_name(raw):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def build_meeting(meeting_id, record, sources):
    mix_path = AUDIO_DIR / f"{meeting_id}.wav"
    mixture, sample_rate = read_mono(mix_path)
    if sample_rate != 16000:
        raise RuntimeError(f"Unexpected sample rate for {meeting_id}: {sample_rate}")
    n_samples = len(mixture)

    if abs(record["duration"] - n_samples / sample_rate) > 0.02:
        raise RuntimeError(
            f"Duration mismatch for {meeting_id}: manifest={record['duration']} "
            f"audio={n_samples / sample_rate}"
        )

    stems = {}
    segments = {}
    expected_rms = {}
    for supervision in record["supervisions"]:
        speaker = supervision["speaker"]
        custom = supervision["custom"]
        prefix = custom["original_cut_id"].rsplit("-", 1)[0]
        source = sources[prefix]
        cut_audio, cut_rate = read_mono(source["source"])
        if cut_rate != sample_rate:
            raise RuntimeError(f"Unexpected source rate: {source['source']}")

        if abs(len(cut_audio) / sample_rate - supervision["duration"]) > 0.02:
            raise RuntimeError(
                f"Source duration mismatch for {meeting_id} {speaker}: "
                f"source={len(cut_audio) / sample_rate} sup={supervision['duration']}"
            )

        audio = cut_audio - np.mean(cut_audio, dtype=np.float32)
        target_rms = 10.0 ** (float(custom["speech_level_db"]) / 20.0)
        current_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)

        offset = int(supervision["start"] * sample_rate)
        end = offset + len(audio)
        if end > n_samples:
            audio = audio[: max(0, n_samples - offset)]
            end = n_samples
        if end > len(mixture):
            raise RuntimeError(f"Stem overruns mixture: {meeting_id} {speaker}")

        stems.setdefault(speaker, np.zeros(n_samples, dtype=np.float64))
        stems[speaker][offset:end] += audio
        segments.setdefault(speaker, []).append(
            {
                "start": float(supervision["start"]),
                "duration": float(supervision["duration"]),
                "speech_level_db": float(custom["speech_level_db"]),
                "source_cut_id": source["cut_id"],
            }
        )
        expected_rms.setdefault(speaker, []).append(target_rms)

    summed = np.sum(list(stems.values()), axis=0)
    maxval = float(np.amax(np.abs(summed)))
    gain_f = 0.99 / maxval if maxval > 0 else 1.0
    gain_applied = maxval > 0.99
    if gain_applied:
        summed = summed * gain_f

    residual = summed - mixture.astype(np.float64)
    signal_power = float(np.mean(np.square(mixture, dtype=np.float64)))
    noise_power = float(np.mean(np.square(residual)))
    snr = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    correlation = float(np.corrcoef(summed, mixture)[0, 1])
    max_abs_residual = float(np.max(np.abs(residual)))

    meeting_dir = SOLO_DIR / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    speakers = []
    for speaker in sorted(stems):
        target = stems[speaker]
        if gain_applied:
            target = target * gain_f
        filename = f"{safe_name(speaker)}.wav"
        sf.write(meeting_dir / filename, target, sample_rate, subtype="PCM_16")
        speech = sum(item["duration"] for item in segments[speaker])
        speakers.append(
            {
                "raw": speaker,
                "audio": (meeting_dir / filename).relative_to(DEMO).as_posix(),
                "speech": speech,
                "turns": len(segments[speaker]),
                "rms_db": rms_db(target),
                "peak": float(np.max(np.abs(target))),
                "segments": segments[speaker],
            }
        )

    return {
        "duration": n_samples / sample_rate,
        "sample_rate": sample_rate,
        "audio": mix_path.relative_to(DEMO).as_posix(),
        "global_gain_applied": gain_applied,
        "global_gain": gain_f,
        "verification": {
            "snr_db": snr,
            "correlation": correlation,
            "max_abs_residual": max_abs_residual,
            "mix_peak": float(np.max(np.abs(mixture))),
        },
        "speakers": speakers,
    }


def main():
    meeting_ids = load_meeting_ids()
    records = load_demo_records(meeting_ids)
    sources = load_source_cuts(needed_cut_prefixes(records))
    SOLO_DIR.mkdir(parents=True, exist_ok=True)

    meetings = {}
    for meeting_id in meeting_ids:
        result = build_meeting(meeting_id, records[meeting_id], sources)
        meetings[meeting_id] = result
        verification = result["verification"]
        print(
            f"{meeting_id}  speakers={len(result['speakers'])}  "
            f"snr={verification['snr_db']:.1f} dB  "
            f"max_residual={verification['max_abs_residual']:.2e}  "
            f"gain={'yes' if result['global_gain_applied'] else 'no'}"
        )

    payload = {
        "source": "FastMSS synthetic meetings; exact source stems",
        "simulation": {
            "reverberate": False,
            "use_fir": False,
            "add_noise": False,
        },
        "meetings": meetings,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

