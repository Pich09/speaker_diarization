"""Approximate model-based target-speaker extraction for arbitrary audio.

This is the model-based counterpart to the exact source-stem solo tracks used
by the synthetic demo.  It chunks the mixture, runs a SpeechBrain SepFormer
separator, selects the output whose ECAPA embedding is closest to an enrollment
clip, and stitches the selected chunks back together.

It is not exact.  Separation quality drops with overlapping speech, more than
three active speakers, noise, and domain mismatch.  For 2-speaker input the
WHAMR 2-source model is used; for 3+ speakers the Libri3Mix model is used.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_2SPK = "speechbrain/sepformer-whamr16k"
DEFAULT_3SPK = "speechbrain/sepformer-libri3mix"
DEFAULT_EMBEDDING = "speechbrain/spkrec-ecapa-voxceleb"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", required=True, type=Path, help="Mixed audio file.")
    parser.add_argument("--enroll", required=True, type=Path, help="Clean enrollment clip.")
    parser.add_argument("--out", required=True, type=Path, help="Output target-only WAV.")
    parser.add_argument("--rttm", type=Path, help="Optional target RTTM to restrict processing.")
    parser.add_argument("--speaker", help="Target speaker label in --rttm.")
    parser.add_argument("--num-speakers", type=int, default=2, help="2 uses WHAMR; 3+ uses Libri3Mix.")
    parser.add_argument("--model", help="Override the SpeechBrain separation model.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--savedir", type=Path, help="Model cache directory.")
    parser.add_argument("--embedding-savedir", type=Path, default=ROOT / "models" / "spkrec-ecapa-voxceleb")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.30, help="Minimum enrollment cosine similarity.")
    parser.add_argument("--passes", type=int, default=1, help="Re-separate the selected source to peel more speakers.")
    parser.add_argument("--margin", type=float, default=0.25, help="RTTM activity margin in seconds.")
    parser.add_argument("--no-level-match", action="store_true", help="Do not match target level to the mix.")
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def read_mono(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio[:, 0], sample_rate


def active_clip(audio, sample_rate, seconds=5.0):
    length = int(seconds * sample_rate)
    if len(audio) <= length:
        return audio
    window = np.ones(length, dtype=np.float64) / length
    energy = np.convolve(np.square(audio, dtype=np.float64), window, mode="valid")
    start = int(np.argmax(energy))
    return audio[start : start + length]


def parse_rttm(path, target):
    intervals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        raw = parts[-2] if parts[-2] != "<NA>" else parts[-3]
        if raw != target:
            continue
        start = float(parts[3])
        intervals.append((start, start + float(parts[4])))
    if not intervals:
        raise RuntimeError(f"No intervals for speaker {target!r} in {path}")
    return sorted(intervals)


def intervals_to_mask(intervals, n_samples, sample_rate):
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in intervals:
        mask[max(0, int(start * sample_rate)) : min(n_samples, int(end * sample_rate))] = True
    return mask


def cosine(a, b):
    return float(torch.dot(a, b))


def load_models(args, device, model_source, savedir):
    sep = SepformerSeparation.from_hparams(
        source=model_source,
        savedir=str(savedir),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": device},
    )
    enc = EncoderClassifier.from_hparams(
        source=args.embedding_model,
        savedir=str(args.embedding_savedir),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": device},
    )
    return sep, enc


def embedding(enc, audio, device):
    wav = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        vector = enc.encode_batch(wav).squeeze()
    return vector / vector.norm()


def select_source(sep, enc, enrollment, chunk, device, passes, threshold):
    wav = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
    best_source = None
    best_similarity = -2.0
    for _ in range(max(1, passes)):
        with torch.no_grad():
            sources = sep.separate_batch(wav)[0].cpu().numpy()
        candidate = None
        candidate_similarity = -2.0
        for index in range(sources.shape[1]):
            source = sources[:, index]
            similarity = cosine(enrollment, embedding(enc, source, device))
            if similarity > candidate_similarity:
                candidate = source
                candidate_similarity = similarity
        if candidate_similarity <= best_similarity + 0.005:
            break
        best_source = candidate
        best_similarity = candidate_similarity
        wav = torch.tensor(candidate, dtype=torch.float32, device=device).unsqueeze(0)
    if best_source is None or best_similarity < threshold:
        return None, best_similarity
    return best_source, best_similarity


def main():
    args = parse_args()
    if args.rttm and not args.speaker:
        raise SystemExit("--speaker is required with --rttm")

    device = resolve_device(args.device)
    model_source = args.model or (DEFAULT_3SPK if args.num_speakers >= 3 else DEFAULT_2SPK)
    if args.savedir is None:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_source.split("/")[-1])
        args.savedir = ROOT / "models" / slug

    mix, sample_rate = read_mono(args.mix)
    enroll, enroll_rate = read_mono(args.enroll)
    if enroll_rate != sample_rate:
        raise SystemExit(f"Enrollment rate {enroll_rate} != mix rate {sample_rate}")

    intervals = parse_rttm(args.rttm, args.speaker) if args.rttm else None
    target_mask = intervals_to_mask(intervals, len(mix), sample_rate) if intervals else None
    other_mask = None
    if args.rttm:
        other_mask = np.zeros(len(mix), dtype=bool)
        for line in args.rttm.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            raw = parts[-2] if parts[-2] != "<NA>" else parts[-3]
            if raw == args.speaker:
                continue
            start = float(parts[3])
            end = start + float(parts[4])
            other_mask[max(0, int(start * sample_rate)) : min(len(mix), int(end * sample_rate))] = True

    print(f"device={device} model={model_source} rate={sample_rate} duration={len(mix) / sample_rate:.1f}s")
    sep, enc = load_models(args, device, model_source, args.savedir)
    enrollment = embedding(enc, active_clip(enroll, sample_rate), device)

    chunk_samples = max(int(args.chunk_seconds * sample_rate), sample_rate)
    hop_samples = max(chunk_samples - int(args.overlap_seconds * sample_rate), sample_rate // 2)
    fade_samples = min(int(0.05 * sample_rate), chunk_samples // 2)

    output = np.zeros(len(mix), dtype=np.float64)
    weights = np.zeros(len(mix), dtype=np.float64)
    similarities = []
    selected = 0
    considered = 0

    for start in range(0, len(mix), hop_samples):
        end = min(start + chunk_samples, len(mix))
        if end - start < sample_rate // 2:
            break
        if target_mask is not None:
            begin_time = start / sample_rate - args.margin
            end_time = end / sample_rate + args.margin
            if not any(end_time > a and begin_time < b for a, b in intervals):
                continue
        considered += 1
        source, similarity = select_source(
            sep, enc, enrollment, mix[start:end], device, args.passes, args.threshold
        )
        similarities.append(similarity)
        if source is None:
            continue
        selected += 1
        window = np.ones(len(source), dtype=np.float64)
        if start > 0:
            window[:fade_samples] = np.linspace(0.0, 1.0, fade_samples)
        if end < len(mix):
            window[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
        output[start:end] += source * window
        weights[start:end] += window

    output = output / np.maximum(weights, 1e-9)
    if not np.any(weights):
        raise SystemExit("No chunks were selected; lower --threshold or check the RTTM/enrollment.")

    level_ratio = 1.0
    if target_mask is not None and not args.no_level_match:
        exclusive = target_mask if other_mask is None else (target_mask & ~other_mask)
        if exclusive.sum() < sample_rate // 2:
            exclusive = target_mask
        mix_rms = float(np.sqrt(np.mean(np.square(mix[exclusive], dtype=np.float64))))
        out_rms = float(np.sqrt(np.mean(np.square(output[exclusive], dtype=np.float64))))
        if out_rms > 1e-8:
            level_ratio = mix_rms / out_rms
            output *= level_ratio

    peak = float(np.max(np.abs(output)))
    if peak > 0.99:
        output *= 0.99 / peak

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, output, sample_rate, subtype="PCM_16")

    print(
        f"chunks considered={considered} selected={selected} "
        f"mean_similarity={np.mean(similarities) if similarities else float('nan'):.3f} "
        f"min_similarity={np.min(similarities) if similarities else float('nan'):.3f} "
        f"level_ratio={level_ratio:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

