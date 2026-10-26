# DiariZen Khmer Pilot - Todo List

Last updated: 2026-10-26

## Done

- [x] Build one-shot code in `scripts/`:
  `scripts/prepare_data.py`, `scripts/download_model.py`, `scripts/check_checkpoint.py`,
  `scripts/run_zero_shot.py`, `scripts/score_rttm.py`, `scripts/export_model.py`, `scripts/run_oneshot.py`,
  `configs/finetune_khmer.toml`, `README.md`, `requirements.txt`, `.gitignore`.
- [x] Verify all scripts compile and smoke-test `scripts/prepare_data.py`,
  `scripts/export_model.py`, `scripts/score_rttm.py`, and `scripts/run_oneshot.py --dry-run`.
- [x] Update `docs/KHMER_DIARIZATION_FINETUNE_PLAN.md` with `rttm.txt`
  layout, holdout rules, `ckpt_dir`, checkpoint path, and the one-shot appendix.
- [x] Clone DiariZen (with `dscore` submodule) into `DiariZen`.
- [x] Install DiariZen, pyannote-audio 3.1.1, and dependencies into the
  `fastmss` conda env without downgrading torch/numpy.
- [x] Patch `DiariZen/pyannote-audio/pyannote/audio/tasks/segmentation/mixins.py`
  for torchaudio >= 2.2 (`AudioMetaData` fallback).
- [x] Download the v2 model and embedding to
  `models/diarizen-wavlm-large-s80-md-v2` and `models/embedding-resnet34`.
- [x] Verify checkpoint keys: 599/599 loaded, 0 missing, 0 unexpected.
- [x] Build the smoke-test holdout bundle:
  `python scripts/run_oneshot.py --stages prepare --yes --test-speakers 2 --dev-speakers 2 --max-train-hours 1`
- [x] Run zero-shot baseline on a 2-file slice:
  `python scripts/run_oneshot.py --stages baseline --yes --test-limit 2`
- [x] Score 2-file baseline smoke test: DER 3.39%, collar 0.25.
- [x] Run and score the full 82-file zero-shot baseline: **DER 9.58%**,
  JER 19.95%, B3-F1 0.89 (`runs/scores/baseline_holdout/metrics.json`).
- [x] Patch Windows/runtime issues: `onnxruntime`, torch weights-only load,
  soundfile audio fallback, Perl for dscore, PyTorch embedding path.
- [x] Render frozen-WavLM config: `runs/configs/khmer_finetune_holdout.toml`.
- [x] Generate self-contained HTML baseline report: `report.html`
  (`python scripts/build_report.py`); preserved unchanged.
- [x] Run a 1-epoch VRAM probe: batch 1, grad accumulation 4, 247 chunks,
  no OOM; about 41 s for the epoch.
- [x] Run the initial frozen-WavLM pilot on the 1.03 h bundle: 7 of 20 epochs
  completed before the session was interrupted; best checkpoint at epoch 3
  (validation NLL 0.207).
- [x] Export the best checkpoint and run fine-tuned inference on all 82 test
  files.
- [x] Score the initial pilot: **DER 5.31%**, JER 15.98%, B3-F1 0.93;
  delta 4.27 points vs baseline. Gate PASS.
- [x] Run the full 8-epoch frozen-WavLM fine-tune on the 6,207-meeting /
  117.18 h bundle; best checkpoint epoch 8 (validation NLL 0.1506,
  validation DER 3.10%).
- [x] Export and score all 82 test files on the full run: **DER 4.62%**,
  JER 15.53%, B3-F1 0.96; delta **4.96 points** vs baseline. Gate PASS.
- [x] Rebuild `report_finetuned.html` for `runs_full`; update
  `runs_full/TRAINING_PROGRESS.txt` to COMPLETED. `report.html` unchanged.
- [x] Add exact source-stem solo playback for the 10-meeting synthetic demo:
  29 stems, reconstruction SNR 63-78 dB, residual at the PCM16 floor.
- [x] Add approximate model-based target extraction for arbitrary audio:
  SpeechBrain SepFormer + ECAPA enrollment scoring (`extract_target.py`).
- [x] Reorganize the repository into `scripts/`, `configs/`, and `docs/`;
  add a compile-only GitHub Actions workflow and a GitHub-safe `.gitignore`.

## In Progress

- [ ] None.

## Next

- [ ] Build real-meeting speaker isolation and solo playback:
  - [ ] Ingest a real Khmer meeting WAV plus RTTM.
  - [ ] Run the current SepFormer/ECAPA path and collect per-speaker stems.
  - [ ] Compare WeSep/MossFormer2 target extraction on the same meeting.
  - [ ] Add a manifest-driven player with speaker selector and solo playback.
  - [ ] Evaluate target retention, leakage, intelligibility, and overlap.
- [ ] Evaluate the model-based target extractor on real Khmer conversational
  audio with an independent enrollment clip; report SDR/leakage.
- [ ] Consider a longer schedule only if downstream DER still improves; the
  8-epoch run selected epoch 8 by validation NLL.
- [ ] Expand speaker diversity rather than hours; the 117 h full bundle still
  has only 16 training speakers. Keep the same test split for comparability.

## Later

- [ ] Collect 1-2 h of real Khmer conversational audio for evaluation.
- [ ] Unfreeze WavLM only if the frozen pilot improves DER by >= 1.0 point.
- [ ] Re-check licensing before any release (model CC BY-NC 4.0, corpus CC BY-SA 4.0).

## Environment Notes

- Python: `C:\Users\USer\anaconda3\envs\fastmss\python.exe` (3.11.17).
- torch 2.14.1+cu126, torchaudio 2.11.0+cu126, numpy 2.4.6.
- GPU: RTX 3050 Laptop, 4 GB VRAM (about 3.8 GB free).
- Installed `onnxruntime` and `perl` (conda-forge) for local inference/scoring.
- Skipped `pesq` (needs MSVC build tools) and `onnxruntime-gpu`.
- Embedding is loaded as PyTorch from `models/embedding-resnet34/pytorch_model.bin`;
  the directory name deliberately avoids the substring `wespeaker`, which would
  otherwise select pyannote's ONNX loader.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set by the scripts for trusted HF checkpoints.
- `torchaudio` has no working FFmpeg/torchcodec; DiariZen audio loading falls back to
  `soundfile` for WAV input.
- dscore is invoked with Perl for `md-eval-22.pl` on Windows.



