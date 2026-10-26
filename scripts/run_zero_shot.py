import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "DiariZen", ROOT):
    if (candidate / "diarizen").is_dir():
        sys.path.insert(0, str(candidate))
        break

from diarizen.pipelines.inference import DiariZenPipeline


def load_scp(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec_id, wav = line.split(maxsplit=1)
            rows.append((rec_id, wav))
    return rows


def resolve_embedding(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "pytorch_model.bin"
    if not path.exists():
        raise SystemExit(f"Embedding model not found: {path}")
    return str(path)


def build_config_parse(args):
    if all(
        value is None
        for value in (
            args.seg_duration,
            args.segmentation_step,
            args.batch_size,
            args.min_speakers,
            args.max_speakers,
        )
    ):
        return None
    inference_args = {
        "seg_duration": args.seg_duration if args.seg_duration is not None else 16,
        "segmentation_step": args.segmentation_step if args.segmentation_step is not None else 0.1,
        "batch_size": args.batch_size if args.batch_size is not None else 32,
        "apply_median_filtering": True,
    }
    clustering_args = {
        "method": "VBxClustering",
        "min_speakers": args.min_speakers if args.min_speakers is not None else 1,
        "max_speakers": args.max_speakers if args.max_speakers is not None else 20,
        "ahc_criterion": "distance",
        "ahc_threshold": 0.6,
        "Fa": 0.07,
        "Fb": 0.8,
        "lda_dim": 128,
        "max_iters": 20,
    }
    return {"inference": {"args": inference_args}, "clustering": {"args": clustering_args}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--scp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seg-duration", type=int, default=None)
    parser.add_argument("--segmentation-step", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "config.toml").exists():
        raise SystemExit(f"Missing config.toml in {model_dir}")
    if not (model_dir / "pytorch_model.bin").exists():
        raise SystemExit(f"Missing pytorch_model.bin in {model_dir}")
    if not (model_dir / "plda").exists():
        raise SystemExit(f"Missing plda/ in {model_dir}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_scp(args.scp)
    if args.limit:
        rows = rows[: args.limit]

    pipeline = DiariZenPipeline(
        diarizen_hub=model_dir,
        embedding_model=resolve_embedding(args.embedding_model),
        config_parse=build_config_parse(args),
        rttm_out_dir=str(out_dir),
    )

    count = 0
    for rec_id, wav in rows:
        target = out_dir / f"{rec_id}.rttm"
        if args.skip_existing and target.exists():
            print(f"SKIP {rec_id}")
            continue
        if not Path(wav).exists():
            print(f"MISSING {rec_id} {wav}")
            continue
        print(f"DIARIZE {rec_id}")
        pipeline(wav, sess_name=rec_id)
        count += 1
    print(f"Wrote {count} RTTM files to {out_dir}")


if __name__ == "__main__":
    main()

