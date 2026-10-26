import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "DiariZen", ROOT):
    if (candidate / "diarizen").is_dir():
        sys.path.insert(0, str(candidate))
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    import torch

    from diarizen.models.eend.model_wavlm_conformer import Model

    model_dir = Path(args.model_dir).expanduser().resolve()
    ckpt_path = model_dir / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    model = Model(
        wavlm_src="wavlm_large_s80_md",
        wavlm_layer_num=25,
        wavlm_feat_dim=1024,
        attention_in=256,
        ffn_hidden=1024,
        num_head=4,
        num_layer=4,
        kernel_size=31,
        dropout=0.1,
        use_posi=False,
        output_activate_function=False,
        max_speakers_per_chunk=4,
        max_speakers_per_frame=4,
        chunk_size=16,
        selected_channel=0,
    )
    target = model.state_dict()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and not any(
        key.startswith("wavlm_model.") for key in state
    ):
        state = state["state_dict"]
    missing = [key for key in target if key not in state]
    unexpected = [key for key in state if key not in target]
    loaded = len(target) - len(missing)
    print(f"target={len(target)} loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("MISSING", missing[:20])
    if unexpected:
        print("UNEXPECTED", unexpected[:20])
    if loaded < len(target) * 0.9:
        raise SystemExit("Less than 90% of model weights matched the checkpoint")
    print("CHECK OK")


if __name__ == "__main__":
    main()

