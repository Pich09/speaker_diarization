import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_full"
RUN_LOG = RUNS / "run_8epoch.log"
ERR_LOG = RUNS / "run_8epoch.err.log"
OUT = RUNS / "TRAINING_PROGRESS.txt"
PID_FILE = RUNS / "run_8epoch.pid"
TEST_SCP = ROOT / "data" / "diarizen" / "khmer_full" / "holdout" / "test" / "wav.scp"
EXP_DIR = RUNS / "exp" / "khmer_finetune_holdout"
CKPT_DIR = EXP_DIR / "checkpoints"
BASELINE_DIR = RUNS / "zero_shot" / "holdout"
FINETUNED_DIR = RUNS / "finetuned" / "khmer_finetune_holdout"

TS_RE = re.compile(r"^(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+")
EPOCH_RE = re.compile(r"Epoch (\d+) out of (\d+)")
TRAIN_LOSS_RE = re.compile(r"Training Loss 'Loss' on epoch (\d+): ([0-9.]+)")
VAL_RE = re.compile(r"Validation Loss/DER on epoch (\d+): ([0-9.]+) / ([0-9.]+)")
BEST_RE = re.compile(r"Found new best score: ([0-9.]+)")
NO_IMPROVE_RE = re.compile(r"Score did not improve from ([0-9.]+) at epoch (\d+)")
PATIENCE_RE = re.compile(r"Early stopping counter: (\d+) out of (\d+)")
TQDM_RE = re.compile(r"(\d+)%\|+\s*(\d+)/(\d+) \[([0-9:]+)<([0-9:]+)")
PHASE_EVENTS = (
    "========= Epoch",
    "Training Loss 'Loss' on epoch",
    "Begin validation",
    "Validation inference finished",
    "Validation Loss/DER on epoch",
    "Found new best score",
    "Score did not improve",
    "Early stopping counter",
)


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_tail(path, max_bytes=400_000):
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def trainer_logs():
    return sorted(EXP_DIR.glob("khmer_finetune_holdout_*.log"), key=lambda path: path.stat().st_mtime)


def parse_ts(line):
    match = TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(f"{datetime.now().year}-{match.group(1)}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def count_files(path, pattern="*.rttm"):
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [item.strip() for item in result.stdout.strip().split(",")]
        if len(values) == 3:
            return f"{values[0]} MiB used / {values[1]} MiB free / {values[2]}% util"
    except Exception:
        pass
    return "n/a"


def fmt_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_hms(text):
    parts = [int(part) for part in text.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def progress_bar(fraction, width=28):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {fraction * 100:5.1f}%"


def parse_trainer(lines):
    data = {
        "max_epochs": 8,
        "starts": {},
        "train_loss": {},
        "val": {},
        "best": None,
        "best_time": None,
        "patience": None,
        "events": [],
    }
    for line in lines:
        ts = parse_ts(line)
        match = EPOCH_RE.search(line)
        if match and "=========" in line:
            epoch = int(match.group(1))
            data["max_epochs"] = int(match.group(2))
            is_new = epoch not in data["starts"]
            data["starts"][epoch] = ts
            if is_new:
                data["events"].append((ts, f"epoch {epoch}/{data['max_epochs']} started"))
            continue
        match = TRAIN_LOSS_RE.search(line)
        if match:
            epoch = int(match.group(1))
            is_new = epoch not in data["train_loss"]
            data["train_loss"][epoch] = (float(match.group(2)), ts)
            if is_new:
                data["events"].append((ts, f"epoch {epoch} training loss {match.group(2)}"))
            continue
        match = VAL_RE.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            der = float(match.group(3))
            is_new = epoch not in data["val"]
            data["val"][epoch] = (loss, der, ts)
            if is_new:
                data["events"].append((ts, f"epoch {epoch} validation loss {loss:.4f}, DER {der * 100:.2f}%"))
            continue
        match = BEST_RE.search(line)
        if match:
            data["best"] = float(match.group(1))
            data["best_time"] = ts
            data["events"].append((ts, f"new best validation loss {match.group(1)}"))
            continue
        match = NO_IMPROVE_RE.search(line)
        if match:
            data["events"].append((ts, f"epoch {match.group(2)} no improvement (best {match.group(1)})"))
            continue
        match = PATIENCE_RE.search(line)
        if match:
            data["patience"] = (int(match.group(1)), int(match.group(2)), ts)
            data["events"].append((ts, f"early stopping counter {match.group(1)}/{match.group(2)}"))
            continue
        if "Begin validation" in line:
            data["events"].append((ts, "validation started"))
        elif "Validation inference finished" in line:
            data["events"].append((ts, "validation inference finished"))
        elif "Saving current state to" in line and "best" in line:
            data["events"].append((ts, "best checkpoint saved"))
    return data


def last_tqdm(text):
    matches = TQDM_RE.findall(text)
    if not matches:
        return None
    percent, done, total, elapsed, remaining = matches[-1]
    return {
        "percent": int(percent),
        "done": int(done),
        "total": int(total),
        "elapsed": elapsed,
        "remaining": remaining,
    }


def collect_events(events, limit=16):
    out = []
    for ts, text in events[-limit:]:
        stamp = ts.strftime("%H:%M:%S") if ts else "--:--:--"
        out.append(f"  {stamp}  {text}")
    return out or ["  No training events yet."]


def build_report(pid):
    now = datetime.now()
    run_text = read_text(RUN_LOG)
    run_lines = run_text.splitlines()
    logs = trainer_logs()
    trainer_lines = []
    for path in logs:
        trainer_lines.extend(read_text(path).splitlines())
    err_tail = read_tail(ERR_LOG)
    trainer = parse_trainer(trainer_lines)
    tqdm = last_tqdm(err_tail)

    baseline_count = count_files(BASELINE_DIR)
    finetuned_count = count_files(FINETUNED_DIR)
    total_test = sum(1 for line in read_text(TEST_SCP).splitlines() if line.strip())

    stage = "STARTING"
    for line in run_lines:
        if not line.startswith("RUN "):
            continue
        if "prepare_data.py" in line:
            stage = "PREPARING DATA"
        elif "run_zero_shot.py" in line:
            if "runs_full/zero_shot" in line or "runs_full\\zero_shot" in line:
                stage = "BASELINE INFERENCE"
            else:
                stage = "FINE-TUNED INFERENCE"
        elif "accelerate" in line:
            stage = "TRAINING"
        elif "export_model.py" in line:
            stage = "EXPORTING MODEL"
        elif "score_rttm.py" in line:
            stage = "SCORING"
    if "=== GATE SUMMARY ===" in run_text:
        stage = "FINISHED"

    try:
        process = psutil.Process(pid)
        elapsed = now.timestamp() - process.create_time()
        alive = True
        process_line = f"PID {pid} | alive | elapsed {fmt_duration(elapsed)}"
    except psutil.Error:
        alive = False
        elapsed = 0
        process_line = f"PID {pid} | stopped"

    started = sorted(trainer["starts"])
    validated = sorted(trainer["val"])
    pending = [epoch for epoch in started if epoch not in trainer["val"]]
    current_epoch = pending[-1] if pending else (started[-1] if started else 0)
    max_epochs = trainer["max_epochs"]
    phase = "STARTING"
    if current_epoch:
        if current_epoch in trainer["val"]:
            phase = "BETWEEN EPOCHS"
        elif current_epoch in trainer["train_loss"]:
            phase = "VALIDATION"
        else:
            phase = "TRAINING"

    durations = []
    for epoch in validated:
        start = trainer["starts"].get(epoch)
        end = trainer["val"][epoch][2]
        if start and end:
            durations.append((end - start).total_seconds())
    epoch_duration = statistics.median(durations) if durations else 76 * 60
    epoch_duration = max(epoch_duration, 20 * 60)

    status = stage
    progress_line = progress_bar(0)
    eta_line = ""
    training_remaining = None

    if stage == "BASELINE INFERENCE":
        fraction = baseline_count / total_test if total_test else 0
        status = f"BASELINE INFERENCE ({baseline_count}/{total_test} files)"
        progress_line = progress_bar(fraction)
        eta_line = "Baseline comparison run; training starts when it finishes."
    elif stage == "TRAINING":
        if current_epoch:
            epoch_start = trainer["starts"].get(current_epoch)
            epoch_elapsed = (now - epoch_start).total_seconds() if epoch_start else 0
            if phase == "TRAINING":
                status = f"TRAINING - EPOCH {current_epoch}/{max_epochs}"
                if tqdm and tqdm["total"] > 1000:
                    fraction = tqdm["done"] / tqdm["total"]
                    step_line = (
                        f"Step {tqdm['done']:,}/{tqdm['total']:,} | "
                        f"step elapsed {tqdm['elapsed']} | step ETA {tqdm['remaining']}"
                    )
                    progress_line = progress_bar(fraction)
                    epoch_remaining = epoch_duration * (1 - fraction)
                else:
                    fraction = epoch_elapsed / epoch_duration
                    step_line = f"Epoch elapsed {fmt_duration(epoch_elapsed)}"
                    progress_line = progress_bar(fraction)
                    epoch_remaining = max(0, epoch_duration - epoch_elapsed)
                training_remaining = epoch_remaining + max(0, max_epochs - current_epoch) * epoch_duration
                eta_line = (
                    f"{step_line} | epoch ETA ~{fmt_duration(epoch_remaining)} | "
                    f"training remaining ~{fmt_duration(training_remaining)} | "
                    f"total incl. export/score ~{fmt_duration(training_remaining + 15 * 60)}"
                )
            elif phase == "VALIDATION":
                status = f"TRAINING - EPOCH {current_epoch}/{max_epochs} - VALIDATION"
                if tqdm:
                    fraction = tqdm["done"] / tqdm["total"] if tqdm["total"] else 0
                    validation_remaining = parse_hms(tqdm["remaining"])
                    progress_line = progress_bar(fraction)
                    training_remaining = validation_remaining + max(0, max_epochs - current_epoch) * epoch_duration
                    eta_line = (
                        f"Validation {tqdm['done']:,}/{tqdm['total']:,} | "
                        f"validation ETA ~{fmt_duration(validation_remaining)} | "
                        f"training remaining ~{fmt_duration(training_remaining)} | "
                        f"total incl. export/score ~{fmt_duration(training_remaining + 15 * 60)}"
                    )
                else:
                    progress_line = progress_bar(0.5)
                    training_remaining = epoch_duration * 0.5 + max(0, max_epochs - current_epoch) * epoch_duration
                    eta_line = (
                        f"Validation on the 11.5 h dev set takes ~4.5 min. "
                        f"Training remaining ~{fmt_duration(training_remaining)}."
                    )
            else:
                status = f"TRAINING - EPOCH {current_epoch}/{max_epochs} - TRANSITION"
                progress_line = progress_bar(1)
                eta_line = f"Epoch {current_epoch} complete; preparing next epoch."
        else:
            status = "TRAINING - STARTING"
            eta_line = "Waiting for the first epoch to start."
    elif stage == "FINE-TUNED INFERENCE":
        fraction = finetuned_count / total_test if total_test else 0
        status = f"FINE-TUNED INFERENCE ({finetuned_count}/{total_test} files)"
        progress_line = progress_bar(fraction)
        eta_line = "Training is complete; inference and scoring follow."
    elif stage == "EXPORTING MODEL":
        status = "EXPORTING BEST CHECKPOINT"
        progress_line = progress_bar(1)
        eta_line = "Inference and scoring follow."
    elif stage == "SCORING":
        status = "SCORING"
        progress_line = progress_bar(1)
        eta_line = "Almost done."
    elif stage == "FINISHED":
        status = "FINISHED"
        progress_line = progress_bar(1)
        eta_line = "Gate summary has been written to the run log."
    else:
        eta_line = "Waiting for the orchestrator."

    if not alive and stage != "FINISHED":
        status = "STOPPED OR FAILED"
        eta_line = "The orchestrator process is no longer running. Check run_8epoch.err.log."

    checkpoint_names = []
    if CKPT_DIR.exists():
        checkpoint_names = sorted(path.name for path in CKPT_DIR.iterdir() if path.is_dir())
    recent_checkpoints = ", ".join(checkpoint_names[-5:]) if checkpoint_names else "none yet"

    epoch_lines = []
    all_epochs = sorted(set(trainer["starts"]) | set(trainer["train_loss"]) | set(trainer["val"]))
    for epoch in all_epochs:
        parts = [f"Epoch {epoch}"]
        if epoch in trainer["train_loss"]:
            parts.append(f"train loss {trainer['train_loss'][epoch][0]:.5f}")
        else:
            parts.append("train loss pending")
        if epoch in trainer["val"]:
            loss, der, _ = trainer["val"][epoch]
            parts.append(f"val loss {loss:.5f}")
            parts.append(f"val DER {der * 100:.2f}%")
            if trainer["best"] is not None and abs(loss - trainer["best"]) < 1e-9:
                parts.append("BEST")
        else:
            parts.append("validation pending")
        epoch_lines.append("  " + " | ".join(parts))
    if not epoch_lines:
        epoch_lines = ["  No completed epochs yet."]

    metric_lines = [
        f"best validation loss: {trainer['best']:.5f}" if trainer["best"] is not None else "best validation loss: pending",
        f"max epochs: {max_epochs} | current epoch: {current_epoch or '--'} | phase: {phase}",
    ]
    if trainer["patience"]:
        metric_lines.append(f"early stopping counter: {trainer['patience'][0]}/{trainer['patience'][1]}")
    if logs:
        metric_lines.append(f"trainer logs: {', '.join(path.name for path in logs)}")

    report = f"""DIARIZEN KHMER FINE-TUNE - 8 EPOCH RUN
Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}
Run dir: {RUNS}
Process: {process_line}
Data: 6,207 train meetings / 117.18 h; 611 dev / 11.50 h; 82 test / 1.70 h
Config: 8 epochs max, patience 8, batch 1, grad accum 4, val batch 2, frozen WavLM
GPU: {gpu_status()}

STATUS: {status}
{progress_line}
{eta_line}

EPOCH SUMMARY
""" + "\n".join(epoch_lines) + """

METRICS
""" + "\n".join(f"  {line}" for line in metric_lines) + f"""
Checkpoints: {recent_checkpoints}

RECENT EVENTS
""" + "\n".join(collect_events(trainer["events"])) + """

HOW TO WATCH
  Get-Content runs_full\\TRAINING_PROGRESS.txt -Wait
  This file refreshes every 20 seconds.
  Full trainer log: runs_full\\exp\\khmer_finetune_holdout\\
"""
    return report


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    while True:
        try:
            report = build_report(pid)
            OUT.write_text(report, encoding="utf-8")
            if "=== GATE SUMMARY ===" in read_text(RUN_LOG):
                break
            if not psutil.pid_exists(pid):
                break
        except Exception as exc:
            OUT.write_text(f"Progress monitor error at {datetime.now():%Y-%m-%d %H:%M:%S}\n{exc}\n", encoding="utf-8")
        time.sleep(20)


if __name__ == "__main__":
    main()

