import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = ("download", "check", "prepare", "render", "baseline", "train", "export", "score")


def build_bundles(args):
    if args.mode == "holdout":
        return ["holdout"]
    return [f"fold_{index}" for index in range(args.folds)]


def run(cmd, cwd=None, env=None, dry_run=False):
    print("RUN", " ".join(str(item) for item in cmd))
    if dry_run:
        return 0
    process = subprocess.run([str(item) for item in cmd], cwd=cwd, env=env)
    if process.returncode != 0:
        raise SystemExit(f"Command failed with exit code {process.returncode}: {cmd}")
    return process.returncode


def require(path, message, dry_run=False):
    if dry_run:
        return
    if not Path(path).exists():
        raise SystemExit(message)


def render_config(args, bundle):
    template = Path(args.config_template).read_text(encoding="utf-8")
    bundle_dir = Path(args.data_root) / bundle
    replacements = {
        "__SAVE_DIR__": Path(args.exp_save_dir).as_posix(),
        "__CKPT_DIR__": (Path(args.model_dir) / "pytorch_model.bin").as_posix(),
        "__TRAIN_WAV__": (bundle_dir / "train" / "wav.scp").as_posix(),
        "__TRAIN_RTTM__": (bundle_dir / "train" / "rttm.txt").as_posix(),
        "__TRAIN_UEM__": (bundle_dir / "train" / "all.uem").as_posix(),
        "__DEV_WAV__": (bundle_dir / "dev" / "wav.scp").as_posix(),
        "__DEV_RTTM__": (bundle_dir / "dev" / "rttm.txt").as_posix(),
        "__DEV_UEM__": (bundle_dir / "dev" / "all.uem").as_posix(),
        "__FREEZE__": "true" if not args.unfreeze_wavlm else "false",
        "__MAX_EPOCHS__": str(args.max_epochs),
        "__TRAIN_BATCH__": str(args.train_batch_size),
        "__VAL_BATCH__": str(args.val_batch_size),
        "__NUM_WORKERS__": str(args.num_workers),
        "__GRAD_ACCUM__": str(args.grad_accum),
        "__SEED__": str(args.seed),
        "__CHUNK_SIZE__": str(args.chunk_size),
        "__TRAIN_SHIFT__": str(args.train_shift),
        "__VAL_SHIFT__": str(args.val_shift),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    out = Path(args.configs_dir) / f"khmer_finetune_{bundle}.toml"
    if args.dry_run:
        print(f"RENDER {out}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(template, encoding="utf-8")
        print(f"RENDERED {out}")
    return out


def stage_download(args):
    run(
        [
            args.python,
            str(SCRIPT_DIR / "download_model.py"),
            "--model-id",
            args.model_id,
            "--embedding-repo",
            args.embedding_repo,
            "--out",
            str(args.models_dir),
        ],
        dry_run=args.dry_run,
    )


def stage_check(args):
    require(args.model_dir, f"Model not found: {args.model_dir}. Run the download stage first.", args.dry_run)
    run(
        [
            args.python,
            str(SCRIPT_DIR / "check_checkpoint.py"),
            "--model-dir",
            str(args.model_dir),
        ],
        dry_run=args.dry_run,
    )


def stage_prepare(args):
    run(
        [
            args.python,
            str(SCRIPT_DIR / "prepare_data.py"),
            "--manifest-dir",
            str(Path(args.fastmss_root) / "data" / "sim" / "manifests"),
            "--sim-root",
            str(Path(args.fastmss_root) / "data" / "sim"),
            "--out",
            str(args.data_root),
            "--mode",
            args.mode,
            "--folds",
            str(args.folds),
            "--test-speakers",
            str(args.test_speakers),
            "--dev-speakers",
            str(args.dev_speakers),
            "--max-train-hours",
            str(args.max_train_hours),
            "--seed",
            str(args.seed),
        ],
        dry_run=args.dry_run,
    )


def stage_baseline(args, bundles):
    require(args.model_dir, f"Model not found: {args.model_dir}. Run the download stage first.", args.dry_run)
    for bundle in bundles:
        scp = Path(args.data_root) / bundle / "test" / "wav.scp"
        require(scp, f"Missing test wav.scp: {scp}", args.dry_run)
        out = Path(args.zero_shot_dir) / bundle
        run(
            [
                args.python,
                str(SCRIPT_DIR / "run_zero_shot.py"),
                "--model-dir",
                str(args.model_dir),
                "--embedding-model",
                str(Path(args.embedding_dir) / "pytorch_model.bin"),
                "--scp",
                str(scp),
                "--out",
                str(out),
                "--limit",
                str(args.test_limit),
                "--batch-size",
                str(args.infer_batch_size),
                "--skip-existing",
            ],
            dry_run=args.dry_run,
        )


def stage_train(args, bundles):
    require(args.model_dir, f"Model not found: {args.model_dir}. Run the download stage first.", args.dry_run)
    configs = {bundle: render_config(args, bundle) for bundle in bundles}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.diarizen_dir) + os.pathsep + env.get("PYTHONPATH", "")
    for bundle, config_path in configs.items():
        command = [
            args.accelerate,
            "launch",
            "--num_processes",
            str(args.num_processes),
            "--main_process_port",
            str(args.main_process_port),
            "run_dual_opt.py",
            "-C",
            str(config_path),
            "-M",
            "train",
        ]
        if args.resume:
            command.append("-R")
        run(
            command,
            cwd=Path(args.diarizen_dir) / "recipes" / "diar_ssl",
            env=env,
            dry_run=args.dry_run,
        )


def resolve_best_checkpoint(args, bundle):
    exp_dir = Path(args.exp_save_dir) / f"khmer_finetune_{bundle}" / "checkpoints"
    best = exp_dir / "best"
    if best.exists():
        return best
    candidates = sorted(path for path in exp_dir.glob("epoch_*") if path.is_dir())
    if not candidates:
        raise SystemExit(f"No checkpoints found under {exp_dir}")
    return candidates[-1]


def stage_export(args, bundles):
    require(args.model_dir, f"Model not found: {args.model_dir}. Run the download stage first.", args.dry_run)
    for bundle in bundles:
        if args.dry_run:
            checkpoint = Path(args.exp_save_dir) / f"khmer_finetune_{bundle}" / "checkpoints" / "best"
        else:
            checkpoint = resolve_best_checkpoint(args, bundle)
        out = Path(args.exported_dir) / f"khmer_finetune_{bundle}"
        run(
            [
                args.python,
                str(SCRIPT_DIR / "export_model.py"),
                "--base-model",
                str(args.model_dir),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(out),
                "--embedding-model",
                str(Path(args.embedding_dir) / "pytorch_model.bin"),
            ],
            dry_run=args.dry_run,
        )


def score_split(args, bundle, sys_dir, name):
    bundle_dir = Path(args.data_root) / bundle / "test"
    ref_dir = bundle_dir / "rttm"
    uem = bundle_dir / "all.uem"
    require(ref_dir, f"Missing reference RTTM dir: {ref_dir}", args.dry_run)
    require(Path(sys_dir), f"Missing system RTTM dir: {sys_dir}", args.dry_run)
    cmd = [
        args.python,
        str(SCRIPT_DIR / "score_rttm.py"),
        "--ref-dir",
        str(ref_dir),
        "--sys-dir",
        str(sys_dir),
        "--out",
        str(Path(args.scores_dir) / name),
        "--score-py",
        str(Path(args.diarizen_dir) / "dscore" / "score.py"),
        "--collar",
        str(args.collar),
    ]
    if uem.exists():
        cmd.extend(["--uem", str(uem)])
    if args.ignore_overlaps:
        cmd.append("--ignore-overlaps")
    run(cmd, dry_run=args.dry_run)


def stage_score(args, bundles):
    for bundle in bundles:
        baseline_dir = Path(args.zero_shot_dir) / bundle
        if baseline_dir.exists() and any(baseline_dir.glob("*.rttm")):
            score_split(args, bundle, baseline_dir, f"baseline_{bundle}")
        finetuned_dir = Path(args.finetuned_dir) / f"khmer_finetune_{bundle}"
        if finetuned_dir.exists() and any(finetuned_dir.glob("*.rttm")):
            score_split(args, bundle, finetuned_dir, f"finetuned_{bundle}")


def stage_finetuned_inference(args, bundles):
    for bundle in bundles:
        model_dir = Path(args.exported_dir) / f"khmer_finetune_{bundle}"
        scp = Path(args.data_root) / bundle / "test" / "wav.scp"
        require(model_dir, f"Exported model not found: {model_dir}", args.dry_run)
        require(scp, f"Missing test wav.scp: {scp}", args.dry_run)
        out = Path(args.finetuned_dir) / f"khmer_finetune_{bundle}"
        run(
            [
                args.python,
                str(SCRIPT_DIR / "run_zero_shot.py"),
                "--model-dir",
                str(model_dir),
                "--embedding-model",
                str(Path(args.embedding_dir) / "pytorch_model.bin"),
                "--scp",
                str(scp),
                "--out",
                str(out),
                "--limit",
                str(args.test_limit),
                "--batch-size",
                str(args.infer_batch_size),
                "--skip-existing",
            ],
            dry_run=args.dry_run,
        )


def read_der(path):
    metrics = Path(path) / "metrics.json"
    if not metrics.exists():
        return None
    return json.loads(metrics.read_text(encoding="utf-8")).get("der")


def print_gate(args, bundles):
    print("")
    print("=== GATE SUMMARY ===")
    for bundle in bundles:
        baseline = read_der(Path(args.scores_dir) / f"baseline_{bundle}")
        finetuned = read_der(Path(args.scores_dir) / f"finetuned_{bundle}")
        print(f"{bundle}: baseline DER={baseline} finetuned DER={finetuned}")
        if baseline is not None and finetuned is not None:
            delta = baseline - finetuned
            status = "PASS" if delta >= 1.0 else "FAIL"
            print(f"{bundle}: delta DER={delta:.2f} -> {status}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--diarizen-dir", default=str(ROOT / "DiariZen"))
    parser.add_argument("--fastmss-root", default=str(ROOT.parent / "FastMSS"))
    parser.add_argument("--config-template", default=str(ROOT / "configs" / "finetune_khmer.toml"))
    parser.add_argument("--data-dir", default="data/diarizen/khmer")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-id", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    parser.add_argument("--embedding-repo", default="pyannote/wespeaker-voxceleb-resnet34-LM")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--accelerate", default="accelerate")
    parser.add_argument("--stages", nargs="+", choices=("all",) + STAGES, default=["all"])
    parser.add_argument("--mode", choices=("holdout", "cv"), default="holdout")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-speakers", type=int, default=3)
    parser.add_argument("--dev-speakers", type=int, default=3)
    parser.add_argument("--max-train-hours", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--train-shift", type=int, default=12)
    parser.add_argument("--val-shift", type=int, default=16)
    parser.add_argument("--unfreeze-wavlm", action="store_true")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--main-process-port", type=int, default=1134)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--infer-batch-size", type=int, default=2)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--ignore-overlaps", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    args.diarizen_dir = Path(args.diarizen_dir).expanduser().resolve()
    args.data_root = (workspace / args.data_dir).resolve()
    args.runs_dir = (workspace / args.runs_dir).resolve()
    args.models_dir = (workspace / args.models_dir).resolve()
    args.configs_dir = args.runs_dir / "configs"
    args.exp_save_dir = args.runs_dir / "exp"
    args.zero_shot_dir = args.runs_dir / "zero_shot"
    args.finetuned_dir = args.runs_dir / "finetuned"
    args.scores_dir = args.runs_dir / "scores"
    args.exported_dir = args.runs_dir / "exported"
    args.model_dir = args.models_dir / args.model_id.split("/")[-1]
    args.embedding_dir = args.models_dir / "embedding-resnet34"
    return args


def main():
    args = parse_args()
    if not args.diarizen_dir.is_dir() and not args.dry_run:
        raise SystemExit(
            f"DiariZen not found at {args.diarizen_dir}. Clone it with: "
            "git clone https://github.com/BUTSpeechFIT/DiariZen.git"
        )
    bundles = build_bundles(args)
    stages = list(STAGES) if "all" in args.stages else [stage for stage in args.stages if stage in STAGES]

    print("Plan")
    print(f"  stages      : {stages}")
    print(f"  mode        : {args.mode}")
    print(f"  bundles     : {bundles}")
    print(f"  workspace   : {args.workspace}")
    print(f"  data root   : {args.data_root}")
    print(f"  model       : {args.model_dir}")
    print(f"  freeze WavLM: {not args.unfreeze_wavlm}")
    if not args.yes and not args.dry_run:
        answer = input("Proceed? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    if "download" in stages:
        stage_download(args)
    if "check" in stages:
        stage_check(args)
    if "prepare" in stages:
        stage_prepare(args)
    if "render" in stages:
        for bundle in bundles:
            render_config(args, bundle)
    if "baseline" in stages:
        stage_baseline(args, bundles)
    if "train" in stages:
        stage_train(args, bundles)
    if "export" in stages:
        stage_export(args, bundles)
        stage_finetuned_inference(args, bundles)
    if "score" in stages:
        stage_score(args, bundles)
        print_gate(args, bundles)


if __name__ == "__main__":
    main()





