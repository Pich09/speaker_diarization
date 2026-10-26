# One-Shot Folder: Status and Next Steps

Last updated: 2026-10-26  
Scope: `D:\PROJECTS\Diarizen\oneshot_Diarizen`  
Status: training and scoring complete; no training process is running.

## Executive summary

The one-shot Khmer DiariZen pilot is finished. The full 8-epoch frozen-WavLM
fine-tune passed the gate:

| Item | Value |
|---|---|
| Baseline DER | **9.58%** |
| Fine-tuned DER | **4.62%** |
| Delta | **4.96 points** |
| Gate target | <= 8.58% DER |
| Gate result | **PASS** |
| Best checkpoint | epoch 8 |
| Validation NLL | 0.1506 |
| Validation DER | 3.10% |

The repository is now organized for GitHub, with code in `scripts/`, configs in
`configs/`, documentation in `docs/`, and the playable demo in
`artifacts/dev_demo/`. Large generated files are ignored by Git.

## What has been done

### 1. Data preparation

- Built a speaker-disjoint DiariZen holdout bundle from the FastMSS simulated
  Khmer meetings.
- Full bundle sizes:
  - train: **6,207 meetings / 117.18 h**
  - dev: **611 meetings / 11.50 h**
  - test: **82 meetings / 1.70 h**
- Training set contains **16 speakers**; dev has 17 and test has 16.
- The bundle includes `wav.scp`, `rttm/`, `rttm.txt`, `all.uem`, and a
  `split.json` leakage report.
- The full run used `data/diarizen/khmer_full/` and `runs_full/`.

### 2. Baseline

- 2-file smoke test: DER **3.39%** (optimistic).
- Full 82-file zero-shot baseline: DER **9.58%**, JER **19.95%**, B3-F1
  **0.89** at collar 0.25.
- Baseline predictions and metrics are in `runs_full/scores/baseline_holdout/`.

### 3. Full fine-tune

- Frozen WavLM, 16 s chunks, 8 epochs, batch 1, gradient accumulation 4,
  validation batch 2, no OOM on the 4 GB RTX 3050.
- Best checkpoint: epoch 8, validation NLL **0.1506**, validation DER
  **3.10%**.
- Full 82-file fine-tuned result: DER **4.62%**, JER **15.53%**, B3-F1
  **0.96**.
- Export, 82-file inference, and scoring all completed.
- Checkpoints are in `runs_full/exp/khmer_finetune_holdout/checkpoints/`.
- Exported model is in `runs_full/exported/khmer_finetune_holdout/`.

### 4. Reports

- `report.html` is the frozen baseline report. It was not modified.
  - SHA-256:
    `B3847EECEC35536CD0ED1955C96A95C01584B076AC965BE29E51DE5AEA78FB6D`
- `report_finetuned.html` compares baseline and fine-tuned results, including
  per-file deltas and the gate decision.
- Both reports contain local links to `runs/` and `runs_full/` outputs that are
  intentionally not committed.

### 5. Playable dev demo

- `artifacts/dev_demo/index.html` contains a 10-meeting, 673.7 s demo.
- On those 10 files:
  - fine-tuned DER: **4.24%**
  - baseline DER: **5.68%**
  - delta: **1.44 points**
- The page includes audio, waveforms, interactive timelines,
  fine-tuned/baseline/reference tabs, and a speaker selector.
- Exact per-speaker solo stems: **29 WAVs**.
- Reconstruction SNR: **63-78 dB** against the mixture.
- Residual is at the 16-bit PCM floor.

The exact stems work because FastMSS simulated these meetings with
`reverberate: false`, `use_fir: false`, and `add_noise: false`. They are not a
general real-meeting solution.

### 6. Model-based speaker extraction

- `scripts/extract_target.py` uses SpeechBrain SepFormer plus ECAPA enrollment
  scoring.
- 2-speaker input uses `speechbrain/sepformer-whamr16k`.
- 3+ speaker input uses `speechbrain/sepformer-libri3mix`.
- It supports chunked extraction, optional RTTM restriction, enrollment
  similarity filtering, level matching, and clipping protection.
- Measured SDR against exact stems:
  - clean 2-speaker case: about **6.3 dB**
  - 3-speaker targets with auto enrollment: **4.31 / 2.44 / 0.24 dB**
  - 3-speaker targets with clean enrollment: **4.31 / 3.87 / 5.45 dB**
- Clean enrollment materially improves the weakest speaker: **0.24 -> 5.45 dB**.

### 7. Real-meeting split wrapper

- `scripts/split_meeting.py` takes a mixture and RTTM, picks an enrollment
  interval per speaker, calls `extract_target.py`, and writes:
  - `<out-dir>/<speaker>.wav`
  - `<out-dir>/manifest.json`
- Auto-enrollment chooses target-only diarization regions.
- `--enroll-dir` accepts clean `<speaker>.wav` clips when no non-overlapping
  region exists.
- Tested end-to-end on the synthetic 3-speaker meeting after the repository
  reorganization.

### 8. Repository cleanup

- Code moved to `scripts/`.
- Training templates moved to `configs/`.
- Plan/TODOs/audit moved to `docs/`.
- Added `.gitignore`, `.gitattributes`, and a compile-only GitHub Actions
  workflow.
- Git repository initialized; nothing has been committed yet.
- Git-ready set: **119 files, about 82 MB**.
- The demo is committed; large generated data is not.

## What is intentionally not committed

| Path | Approx. size | Why |
|---|---:|---|
| `models/` | 0.58 GB | downloaded checkpoints |
| `runs/` + `runs_full/` | ~6 GB | checkpoints, logs, predictions, scores |
| `data/` | 7,668 files | generated DiariZen bundles |
| `DiariZen/` | separate clone | upstream framework |
| fine-tuned checkpoint | 265 MB | exceeds normal Git per-file limits |

The fine-tuned checkpoint is at:

```
runs_full/exp/khmer_finetune_holdout/checkpoints/best/pytorch_model.bin
```

If it needs to be shared, use a **GitHub Release asset** or **Git LFS**. Do not
commit it as a normal Git object.

## Current gaps

- No real Khmer meeting audio or reference RTTM is present in the workspace.
- The model-based separator is approximate and degrades with overlap, noise,
  more than three active speakers, and domain mismatch.
- The training data is clean simulated read speech with only 16-17 speakers.
- No real-meeting solo playback UI exists yet; the current player is the
  synthetic demo only.
- No `LICENSE` has been selected. Code licensing differs from the model
  (CC BY-NC 4.0) and corpus (CC BY-SA 4.0).
- Reports link to local `runs/` outputs that are not committed.

## What to run next

### A. Bootstrap a clean clone

```powershell
git clone <your-repo-url> oneshot_Diarizen
cd oneshot_Diarizen
git clone https://github.com/BUTSpeechFIT/DiariZen.git
pip install -r requirements.txt
python scripts/run_oneshot.py --dry-run
```

`DiariZen` must be installed following its own README, including the
`pyannote-audio` editable install, `torch`, and the `dscore` submodule. Perl is
required for `md-eval-22.pl` on Windows.

### B. Recreate the full data bundle and baseline

```powershell
python scripts/run_oneshot.py --stages download check --yes
python scripts/run_oneshot.py --stages prepare render --yes `
  --data-dir data/diarizen/khmer_full `
  --runs-dir runs_full `
  --max-train-hours 0 `
  --config-template configs/finetune_khmer_8epoch.toml
python scripts/run_oneshot.py --stages baseline score --yes `
  --data-dir data/diarizen/khmer_full `
  --runs-dir runs_full
```

### C. Reproduce the full 8-epoch fine-tune

```powershell
python scripts/run_oneshot.py --stages train export score --yes `
  --data-dir data/diarizen/khmer_full `
  --runs-dir runs_full `
  --config-template configs/finetune_khmer_8epoch.toml `
  --max-epochs 8 `
  --train-batch-size 1 `
  --val-batch-size 2 `
  --grad-accum 4 `
  --num-workers 0
```

Add `--resume` only when continuing an interrupted run.

### D. Rebuild the reports and demo

```powershell
python scripts/build_report.py
python scripts/build_finetune_report.py
python scripts/build_solo_audio.py
python scripts/build_dev_demo.py
```

Do not run `build_report.py` if `report.html` must remain byte-identical.

### E. Evaluate on a real Khmer meeting

First obtain a reference RTTM from diarization. Then split all speakers:

```powershell
python scripts/split_meeting.py `
  --mix real_meeting.wav `
  --rttm real_meeting.rttm `
  --out-dir stems/real_meeting `
  --num-speakers 3
```

If a speaker has no clean non-overlapping region, provide clean enrollment
clips:

```powershell
python scripts/split_meeting.py `
  --mix real_meeting.wav `
  --rttm real_meeting.rttm `
  --out-dir stems/real_meeting `
  --enroll-dir enroll/real_meeting `
  --num-speakers 3
```

The enrollment directory must contain `<speaker>.wav` files using the same
speaker labels as the RTTM.

Then measure leakage/SDR against a manually isolated reference, or run
downstream DER scoring on the separated tracks. This is the main remaining
quality question.

### F. Publish the repository

Add a `LICENSE` first, then:

```powershell
git add .
git commit -m "Initial DiariZen Khmer one-shot pilot"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

The GitHub Actions workflow only runs `python -m compileall -q scripts`; it does
not download models or data.

## Recommended priority

1. Add a `LICENSE` and push the cleaned repository.
2. Build the real-meeting speaker-isolation path:
   ingest a meeting + RTTM, run the current SepFormer baseline, compare
   WeSep/MossFormer2, and add manifest-driven solo playback.
3. Evaluate that path on real Khmer meeting audio with a reference RTTM and
   clean enrollment clips; report leakage and intelligibility.
4. Expand speaker diversity, not just hours; keep the current test split for
   comparability.
5. Consider a longer fine-tune only if downstream DER still improves.
6. Publish the fine-tuned checkpoint as a release asset or LFS object if needed.
