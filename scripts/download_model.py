import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download(repo_id, out_dir, allow_patterns=None):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    parser.add_argument("--embedding-repo", default="pyannote/wespeaker-voxceleb-resnet34-LM")
    parser.add_argument("--out", default="models")
    parser.add_argument("--embedding-name", default="embedding-resnet34")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    if not args.embedding_only:
        model_dir = download(args.model_id, out / args.model_id.split("/")[-1])
        ckpt = model_dir / "pytorch_model.bin"
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        print(f"MODEL_DIR={model_dir}")
        print(f"MODEL_CKPT={ckpt}")
    if not args.model_only:
        embedding_dir = download(args.embedding_repo, out / args.embedding_name)
        embedding_ckpt = embedding_dir / "pytorch_model.bin"
        if not embedding_ckpt.exists():
            raise SystemExit(f"Missing embedding checkpoint: {embedding_ckpt}")
        print(f"EMBEDDING_DIR={embedding_dir}")
        print(f"EMBEDDING_CKPT={embedding_ckpt}")


if __name__ == "__main__":
    main()
