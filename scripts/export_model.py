import argparse
import json
import shutil
from pathlib import Path

import torch


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    candidate = path / "pytorch_model.bin"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Checkpoint not found: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    base_model = Path(args.base_model).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    checkpoint = resolve_checkpoint(args.checkpoint)

    if not base_model.is_dir():
        raise SystemExit(f"Base model directory not found: {base_model}")
    if not (base_model / "config.toml").exists():
        raise SystemExit(f"Base model is missing config.toml: {base_model}")
    if not (base_model / "plda").exists():
        raise SystemExit(f"Base model is missing plda/: {base_model}")
    if out == base_model:
        raise SystemExit("Output must differ from the base model directory")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base_model, out)
    shutil.copy2(checkpoint, out / "pytorch_model.bin")

    state = torch.load(out / "pytorch_model.bin", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise SystemExit("Checkpoint does not contain a state dict")
    print(f"Exported {len(state)} tensor entries to {out}")

    if args.embedding_model:
        embedding = Path(args.embedding_model).expanduser().resolve()
        if embedding.is_dir():
            embedding = embedding / "pytorch_model.bin"
        if not embedding.exists():
            raise SystemExit(f"Embedding model not found: {embedding}")
        (out / "pipeline_kwargs.json").write_text(
            json.dumps({"embedding_model": embedding.as_posix()}, indent=2),
            encoding="utf-8",
        )
        print(f"EMBEDDING_CKPT={embedding}")
    print(f"EXPORTED_MODEL={out}")


if __name__ == "__main__":
    main()
