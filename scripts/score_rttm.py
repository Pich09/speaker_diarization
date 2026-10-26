import argparse
import json
import subprocess
import sys
from pathlib import Path


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--sys-dir", required=True)
    parser.add_argument("--uem", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-py", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--ignore-overlaps", action="store_true")
    parser.add_argument("--n-digits", type=int, default=2)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir).expanduser().resolve()
    sys_dir = Path(args.sys_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_py is None:
        score_py = Path(__file__).resolve().parents[1] / "DiariZen" / "dscore" / "score.py"
    else:
        score_py = Path(args.score_py).expanduser().resolve()
    if not score_py.exists():
        raise SystemExit(f"dscore score.py not found: {score_py}")
    if not ref_dir.is_dir():
        raise SystemExit(f"Reference RTTM dir not found: {ref_dir}")
    if not sys_dir.is_dir():
        raise SystemExit(f"System RTTM dir not found: {sys_dir}")

    ref_files = sorted(ref_dir.glob("*.rttm"))
    sys_files = sorted(sys_dir.glob("*.rttm"))
    if not ref_files:
        raise SystemExit(f"No reference RTTM files found in {ref_dir}")
    if not sys_files:
        raise SystemExit(f"No system RTTM files found in {sys_dir}")

    ref_ids = {path.stem for path in ref_files}
    sys_ids = {path.stem for path in sys_files}
    common_ids = sorted(ref_ids & sys_ids)
    missing_sys = sorted(ref_ids - sys_ids)
    extra_sys = sorted(sys_ids - ref_ids)
    if not common_ids:
        raise SystemExit("No overlapping reference and system RTTM files")
    if missing_sys:
        print(f"WARNING: {len(missing_sys)} reference files have no system RTTM; scoring {len(common_ids)} common files")
    ref_files = [ref_dir / f"{rec_id}.rttm" for rec_id in common_ids]
    sys_files = [sys_dir / f"{rec_id}.rttm" for rec_id in common_ids]

    ref_scp = out_dir / "ref.scp"
    sys_scp = out_dir / "sys.scp"
    ref_scp.write_text("".join(f"{path.as_posix()}\n" for path in ref_files), encoding="utf-8")
    sys_scp.write_text("".join(f"{path.as_posix()}\n" for path in sys_files), encoding="utf-8")

    cmd = [
        args.python,
        str(score_py),
        "-R",
        str(ref_scp),
        "-S",
        str(sys_scp),
        "--collar",
        str(args.collar),
        "--n_digits",
        str(args.n_digits),
    ]
    if args.ignore_overlaps:
        cmd.append("--ignore_overlaps")
    if args.uem is not None:
        uem = Path(args.uem).expanduser().resolve()
        if not uem.exists():
            raise SystemExit(f"UEM not found: {uem}")
        keep = set(common_ids)
        lines = [
            line
            for line in uem.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.split()[0] in keep
        ]
        eval_uem = out_dir / "eval.uem"
        eval_uem.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd.extend(["-u", str(eval_uem)])

    print("RUN", " ".join(cmd))
    process = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "score.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise SystemExit(f"dscore failed with exit code {process.returncode}")

    der = None
    for line in process.stdout.splitlines():
        if "*** OVERALL ***" in line:
            for token in line.split():
                if is_float(token):
                    der = float(token)
                    break
            break
    metrics = {
        "der": der,
        "collar": args.collar,
        "ignore_overlaps": args.ignore_overlaps,
        "num_ref": len(ref_ids),
        "num_sys": len(sys_ids),
        "num_scored": len(common_ids),
        "missing_sys": missing_sys,
        "extra_sys": extra_sys,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(process.stdout)
    print(f"DER={der}")


if __name__ == "__main__":
    main()
