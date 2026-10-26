# DiariZen Khmer Pilot - Plan Snapshot

Objective: gated one-shot pilot to test whether the simulated Khmer data can
improve on the zero-shot `BUT-FIT/diarizen-wavlm-large-s80-md-v2` baseline.

Full audit and phased plan: `docs/KHMER_DIARIZATION_FINETUNE_PLAN.md`.

## Gate

Pilot passes when finetuned DER is at least 1.0 absolute point lower than the
zero-shot baseline on the same speaker-disjoint test split. Passing only
justifies building a real Khmer evaluation set and a larger pilot; it does not
justify release.

## Phases

1. Setup: clone DiariZen, install requirements, patch torchaudio compatibility.
2. Download: fetch v2 model and WeSpeaker embedding.
3. Check: verify checkpoint keys match the training model.
4. Prepare: build validated, speaker-disjoint data bundles.
5. Baseline: run zero-shot v2 on the test split and score with dscore.
6. Train: frozen-WavLM pilot, 16 s chunks, 10-20 h.
7. Export and score: rebuild an inference model and compare DER.
8. Decide: unfreeze WavLM only if the frozen pilot wins.

## Current State

- One-shot code complete in `scripts/`.
- DiariZen cloned with the dscore submodule.
- DiariZen, pyannote-audio 3.1.1, and dependencies installed into the `fastmss` env.
- Compatibility patches applied:
  - `pyannote/audio/tasks/segmentation/mixins.py`: `AudioMetaData` fallback.
  - `pyannote/audio/core/io.py` and `diarizen/pipelines/inference.py`: `soundfile`
    fallback when torchaudio/torchcodec cannot load WAV.
  - `dscore/scorelib/metrics.py`: invoke `md-eval-22.pl` through Perl on Windows.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set for trusted local checkpoints.
- Embedding is loaded from `models/embedding-resnet34/pytorch_model.bin` as a
  PyTorch pyannote model; the path avoids `wespeaker` to prevent ONNX selection.
- `onnxruntime` and Perl installed; `pesq` and `onnxruntime-gpu` skipped.
- Model and embedding downloaded. Checkpoint check: 599/599 keys, 0 missing,
  0 unexpected.
- Full holdout prepared: train 6,207 meetings / 117.176 h, dev 611 / 11.497 h,
  test 82 / 1.699 h; 2 test speakers and 2 dev speakers designated.
- 2-file baseline smoke test: DER 3.39% at collar 0.25.
- Full 82-file zero-shot baseline: **DER 9.58%**, JER 19.95%, B3-F1 0.89.
  Gate target is therefore finetuned DER <= 8.58%.
- Frozen-WavLM config rendered at `runs/configs/khmer_finetune_holdout.toml`.
- VRAM probe completed: batch 1, grad accumulation 4, 247 chunks, no OOM;
  about 41 s per training epoch on the 4 GB GPU.
- Full frozen-WavLM fine-tune completed all 8 epochs on the 6,207-meeting /
  117.18 h bundle. Best checkpoint: epoch 8 by validation NLL loss (0.1506),
  validation DER 3.10%.
- Fine-tuned inference completed on all 82 test files after exporting the best
  checkpoint.
- Fine-tuned score: **DER 4.62%**, JER 15.53%, B3-F1 0.96. Baseline 9.58% ->
  delta 4.96 points. Gate PASS.
- Comparison report generated at `report_finetuned.html`; the original
  `report.html` is unchanged. Training progress is in
  `runs_full/TRAINING_PROGRESS.txt`.
- Exact source-stem solo playback added to the synthetic demo; approximate
  model-based target extraction added for arbitrary audio.

## Future Work: Real-Meeting Speaker Isolation and Solo Playback

Goal: take a real multi-speaker meeting, diarize it, produce an approximate WAV
per speaker, and add a player that can play one speaker at a time.

This is deliberately separate from the synthetic demo. The synthetic solo stems
are exact because FastMSS kept the source cuts separate. A real single-mic
meeting has no source tracks, so this path must use diarization plus separation
and must be presented as approximate.

### Scope

1. **Input**
   - 16 kHz mono WAV, or multi-channel audio if available.
   - Reference RTTM, or generate one with DiariZen/pyannote.
   - Optional clean enrollment clip per speaker.

2. **Diarization**
   - Use the fine-tuned DiariZen model for Khmer, or pyannote.audio as a
     baseline.
   - Preserve overlap regions in the RTTM.
   - Produce stable speaker labels for the meeting.

3. **Enrollment**
   - Prefer user-provided clean clips via `--enroll-dir`.
   - Otherwise pick target-only intervals automatically.
   - Record whether enrollment was automatic or provided in the manifest.

4. **Separation**
   - Start with the current `scripts/split_meeting.py` /
     `scripts/extract_target.py` SepFormer + ECAPA path.
   - Evaluate target-speaker extractors such as WeSep, TD-SpeakerBeam, or
     MossFormer2 on the same real meeting.
   - Handle 2-3 active speakers first; treat 4+ simultaneous speakers as a
     known limitation.
   - Chunk, select the target source, stitch, level-match, and protect against
     clipping.

5. **Outputs**
   - `stems/<meeting>/<speaker>.wav`
   - `stems/<meeting>/manifest.json` with intervals, enrollment source,
     overlap ratio, model name, and confidence.
   - Optional overlap flag per speaker segment.

6. **Solo playback UI**
   - Extend `scripts/build_dev_demo.py` or add a real-meeting player builder.
   - Speaker selector, solo/mute controls, full-mix reference, and timeline.
   - Clear warning that stems are approximate and may contain leakage.
   - Keep the existing synthetic demo unchanged.

7. **Evaluation**
   - Use a real Khmer meeting with a reference RTTM and, where possible,
     manually isolated reference clips.
   - Measure target retention, non-target leakage, intelligibility, and
     speaker-attributed ASR WER.
   - Record overlap ratio and speaker count per meeting.
   - Acceptance is a manual listening pass plus documented leakage; do not claim
     exact separation.

### Phases

1. Add real-meeting ingestion and RTTM validation.
2. Run the current SepFormer path and collect baseline stems.
3. Compare WeSep/MossFormer2 target extraction on the same meeting.
4. Add the solo-playback page and manifest-driven audio loading.
5. Evaluate, document limitations, and decide whether to keep the current
   separator or switch.

### Commands to start later

```powershell
# Split all speakers after diarization has produced an RTTM.
python scripts/split_meeting.py `
  --mix real_meeting.wav `
  --rttm real_meeting.rttm `
  --out-dir stems/real_meeting `
  --num-speakers 3

# Better enrollment when a speaker has no clean solo region.
python scripts/split_meeting.py `
  --mix real_meeting.wav `
  --rttm real_meeting.rttm `
  --out-dir stems/real_meeting `
  --enroll-dir enroll/real_meeting `
  --num-speakers 3
```

### Non-goals

- Exact source recovery from a single mixed microphone.
- Production-grade separation for heavy overlap or 4+ simultaneous speakers.
- Replacing the synthetic demo; the new player is for real meetings.

## Outcome

The full 8-epoch run passes the 1.0-point gate. This justifies a real Khmer
evaluation set and broader speaker coverage. It does not justify a production
claim: the corpus is simulated clean read speech, only 16-17 speakers are
represented, and only the segmentation model was adapted.

## Known Risks

- GPU is a 4 GB RTX 3050 with ~3.8 GB free; the full 8-epoch run completed
  with batch 1 and no OOM.
- The best checkpoint was selected by validation NLL loss, not by downstream
  DER; a longer schedule may find a better epoch.
- `pesq` and `onnxruntime-gpu` were not installed; not needed for the current path.
- Clean read-speech simulations are domain-mismatched with real Khmer meetings.
- Only 16-17 speakers; source-cut overlap between train and test is expected.
- Model is CC BY-NC 4.0 and the corpus is CC BY-SA 4.0.

