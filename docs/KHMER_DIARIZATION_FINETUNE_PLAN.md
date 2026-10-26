# Khmer Diarization Fine-Tuning Plan

Status: Draft for review  
Target model: `BUT-FIT/diarizen-wavlm-large-s80-md-v2`  
Current data: `data/sim` (simulated Khmer meetings)  
Scope: Decide whether the existing data is suitable, then plan a safe fine-tuning and evaluation path.

---

## 1. Executive Summary

The existing data is **format-compatible but domain-incompatible** with DiariZen v2.

- It can be used for a pipeline smoke test, a small acoustic-adaptation pilot, or a data-generation experiment.
- It is **not sufficient as-is** for a robust Khmer diarization model.
- The main problem is not the 130 hours; it is the **20 speakers, read-speech domain, clean acoustics, synthetic turn-taking, and leaky train/dev split**.

**Recommendation:** Do not start a full 118-hour fine-tune yet. First build a small real Khmer evaluation set, convert the data to DiariZen format, repair the splits, and run a zero-shot baseline. Only then run a 10-20 hour pilot fine-tune.

---

## 2. Model Summary

| Item | Value |
|---|---|
| Model | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` |
| Architecture | WavLM-Large + Conformer, structured pruning at 80% |
| WavLM size after pruning | 63.3M parameters |
| Input | 16 kHz, single-channel |
| Chunk size | 16 s |
| Max speakers per chunk | 4 |
| Max speakers per frame | 4 |
| Clustering | VBx + PLDA |
| Embedding model | WeSpeaker VoxCeleb ResNet34 |
| Training data | AMI, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, DIHARD3, RAMC, VoxConverse |
| License | CC BY-NC 4.0 (non-commercial) |

Important: fine-tuning the DiariZen EEND model only adapts the **segmentation** network. The WeSpeaker embedding model and PLDA clustering are separate and remain domain-mismatched unless they are also adapted.

---

## 3. Current Data Audit

### 3.1 What is compatible

| Check | Result |
|---|---|
| Audio format | 16 kHz mono PCM_16 |
| Labels | RTTM exists for every simulated WAV |
| RTTM format | `SPEAKER <rec> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>` |
| Speaker count | 2-4 speakers per meeting, matches model limit |
| Overlap | 20.7% mean; suitable for overlap-aware training |
| Duration | 60-318 s meetings; model uses 16 s chunks |

### 3.2 What is not suitable

| Issue | Evidence | Impact |
|---|---|---|
| Very low speaker diversity | 20 speaker IDs; top 5 speakers â‰ˆ92% of training speech; 9 speakers <1 h | Model can overfit voices instead of learning diarization |
| Read-speech domain | Source is an ASR cultural corpus of ~8 s read sentences | No natural pauses, repairs, interruptions, or conversational dynamics |
| Synthetic meetings | Mean 68 s, ~9.5 turns per meeting | Far from real meetings/calls with dozens or hundreds of turns |
| Clean acoustics | `add_noise: false`, `reverberate: false` in `FastMSS/recipes/khmer.yaml` | Poor match to far-field, noisy, reverberant DiariZen training data |
| Leaky split | All 20 speakers appear in train and dev; 25.5% of dev source utterances also appear in train | Dev scores will be optimistic and not measure generalization |
| Almost no 3-4 speaker overlap | 3-speaker overlap â‰ˆ1.2%; 4-speaker overlap â‰ˆ0% | Model will not learn high-order overlap well |
| Pseudo alignment | One `utt` alignment per cut | Not a DiariZen blocker, but simulated turn timing is less realistic |

### 3.3 Data statistics

- Topics: 69
- Meetings: 6,900
- Supervisions: 65,324
- Duration: 130.37 h
- Train: 6,240 meetings / 117.89 h
- Dev: 660 meetings / 12.48 h
- Speakers per meeting: 2 (2,249), 3 (2,411), 4 (2,240)
- Meeting duration: min 60.0 s, mean 68.0 s, max 317.8 s
- Overlap fraction: mean 20.7%, p95 43.6%

---

## 4. Target Data Requirements

For a credible Khmer diarization fine-tune, aim for:

| Dimension | Minimum pilot | Target |
|---|---|---|
| Audio | 10-20 h | 100+ h |
| Speakers | 20+ | 100+ |
| Speaker-disjoint test | 3-4 speakers | 15-20% of speakers |
| Domain | Some real conversation | Real meetings, calls, interviews, podcasts |
| Acoustics | Some noise/reverb | Far-field and near-field mixture |
| Overlap | Natural overlap | 10-30% overlap, including 3-4 speaker regions |
| Labels | RTTM | RTTM with consistent speaker IDs |
| License | Research-compatible | Compatible with intended use |

The current corpus meets the duration target but fails the speaker, domain, and acoustic-diversity targets.

---

## 5. Proposed Plan

### Phase 0 - Decisions and Guardrails

Tasks:
- Confirm the target domain: real Khmer meetings/calls, or the simulated corpus only.
- Confirm whether the work is research-only or commercial.
- Confirm GPU resources and storage.
- Decide whether to fine-tune the pruned v2 checkpoint or an unpruned DiariZen model and prune afterward.

Deliverables:
- One-page decision record.
- Approved license/use-case note.

Exit criteria:
- Target domain, license, and compute are explicit.

---

### Phase 1 - Build a Real Khmer Evaluation Set

This is the highest-value step. Without a real held-out set, fine-tuning results cannot be trusted.

Tasks:
- Collect 1-2 h of real conversational Khmer audio.
- Annotate speaker turns in RTTM.
- Ensure 3-4 speakers are held out from all training data.
- Include at least some far-field, noisy, and overlapping speech.
- Optionally include 2-3 meeting/call scenarios.

Deliverables:
- `data/diarizen/khmer/test/` with `wav.scp`, `rttm`, `all.uem`.
- A short data card describing source, domain, and annotation rules.

Exit criteria:
- A speaker-disjoint real test set exists and passes RTTM validation.

---

### Phase 2 - Convert Existing Data to DiariZen Layout

DiariZen does not consume the current Lhotse/NeMo manifests directly. It expects:

```text
data/diarizen/khmer/
  train/
    wav.scp
    rttm/          per-meeting RTTMs for dscore
    rttm.txt       concatenated RTTM for DiariZen DiarizationDataset
    all.uem
  dev/
    wav.scp
    rttm/
    rttm.txt
    all.uem
  test/
    wav.scp
    rttm/
    rttm.txt
    all.uem
```

DiariZen's `DiarizationDataset` reads a single concatenated RTTM file, while
`dscore` is easiest to run per-meeting; the conversion writes both forms.

Format examples:

```text
# wav.scp
<rec_id> <absolute_path_to_wav>

# all.uem
<rec_id> 1 0.000 <duration_seconds>

# rttm
SPEAKER <rec_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

Tasks:
- Use the existing WAV paths directly; do not copy 71.6 GB unless required.
- Generate `wav.scp` from `data/sim/manifests/synth-khmer-*.jsonl.gz`.
- Generate `all.uem` from recording durations.
- Concatenate or reference existing per-meeting RTTMs.
- Ensure `rec_id` in `wav.scp`, `all.uem`, and `rttm` is identical.
- Validate every RTTM interval is inside the recording.
- Validate every speaker label is non-empty and consistent.

Deliverables:
- Conversion script.
- Validation report with zero missing WAVs, RTTMs, or UEM rows.

Exit criteria:
- `wav.scp`, `rttm`, and `all.uem` load successfully in DiariZen's `DiarizationDataset`.

---

### Phase 3 - Repair the Split

The current split is by `md5(recording_id)`, which leaks speakers and source utterances.

Tasks:
- Split by speaker first.
- Recommended for 20 speakers: 5-fold speaker-disjoint cross-validation.
- For a single holdout: 14 train / 3 dev / 3 test speakers.
- Drop every meeting that contains a designated dev or test speaker from train.
- Report shared source cuts instead of trying to eliminate them.
- Optionally hold out topics to test topic generalization.
- Report the number of speakers and source cuts per split.

Deliverables:
- Split manifest listing `rec_id`, speakers, source cuts, and split.
- Leakage report: zero designated test/dev speakers in train meetings, plus source-cut overlap counts.

Exit criteria:
- No designated dev/test speaker appears in any train meeting.
- Source-cut overlap is reported. It is expected to be non-zero because a
  synthetic meeting that contains one held-out speaker may also contain train
  speakers, whose source audio then appears in both train and test. This is not
  held-out-speaker leakage, but it does make test scores mildly optimistic.

---

### Phase 4 - Zero-Shot Baseline

Tasks:
- Run the pretrained v2 pipeline on the held-out real and simulated test sets.
- Score with `dscore`.
- Report:
  - DER with collar 0 and 0.25
  - Miss, false alarm, and speaker confusion
  - JER
  - Results by speaker count (2/3/4)
  - Results by overlap condition
- Save predictions and metrics.

Deliverables:
- Baseline metrics table.
- Error analysis of the largest failure modes.

Exit criteria:
- A reproducible zero-shot baseline exists before any fine-tuning.

---

### Phase 5 - Pilot Fine-Tune

Start small. Do not launch a full 118-hour run first.

Tasks:
- Use 10-20 h of the repaired training split.
- Initialize from the v2 checkpoint.
- Freeze WavLM for the first stage; train the Conformer and classifier.
- Then unfreeze WavLM with a low learning rate.
- Use 16 s chunks to match the v2 architecture.
- Use mixed precision and gradient accumulation if GPU memory is limited.
- Validate on the speaker-disjoint dev set.
- Evaluate on the real test set after each epoch.

Suggested config alignment for v2:

```toml
[meta]
save_dir = "<absolute>/runs/exp"
seed = 3407

[finetune]
finetune = true
ckpt_dir = "<model_dir>/pytorch_model.bin"

[model.args]
wavlm_src = "wavlm_large_s80_md"
wavlm_layer_num = 25
wavlm_feat_dim = 1024
attention_in = 256
ffn_hidden = 1024
num_head = 4
num_layer = 4
dropout = 0.1
chunk_size = 16
max_speakers_per_chunk = 4
max_speakers_per_frame = 4
use_posi = false
output_activate_function = false
selected_channel = 0
max_speakers_per_chunk = 4
max_speakers_per_frame = 4
chunk_size = 16
```

`run_dual_opt.py` builds its output directory as `save_dir / exp_id`, where
`exp_id` is the config filename stem. A config named
`khmer_finetune_holdout.toml` therefore writes checkpoints to
`<save_dir>/khmer_finetune_holdout/checkpoints/{best,epoch_XXXX}/pytorch_model.bin`.

Suggested optimizer starting point:

- WavLM learning rate: `1e-5` to `2e-5`
- Head learning rate: `1e-3`
- Warmup: 5-10% of steps
- Early stopping patience: 5-10 epochs
- Gradient clipping: enabled

Deliverables:
- Pilot checkpoint.
- Comparison table: zero-shot vs. frozen-WavLM vs. unfrozen-WavLM.
- Training curves and validation metrics.

Exit criteria:
- Pilot improves DER on the real speaker-disjoint test set by at least 10% relative, with no major regression on overlap regions.

---

### Phase 6 - Full Fine-Tune or Data Expansion

Only proceed if the pilot wins.

Tasks:
- Add more real conversational Khmer data.
- Consider fine-tuning an unpruned DiariZen model, then applying structured pruning, following the paper's recipe.
- Consider adapting the WeSpeaker embedding model or PLDA with in-domain Khmer data.
- Re-run the full evaluation suite.
- Perform a final leakage and license audit.

Deliverables:
- Final checkpoint.
- Model card with training data, license, and limitations.
- Reproducible training and evaluation scripts.

Exit criteria:
- Final model beats the zero-shot baseline on the real speaker-disjoint test set and passes the license review.

---

## 6. Augmentation Strategy

If real data cannot be collected immediately:

- Regenerate FastMSS simulations with `reverberate: true`.
- Add background noise with `add_noise: true` and a noise corpus.
- Keep the current overlap distribution but add more 3-4 speaker overlap.
- Increase speaker diversity if more Khmer speakers can be found.
- Treat this as augmentation only; it does not replace real conversational data.

Do not expect clean read-speech simulations to transfer well to real far-field meetings without these changes.

---

## 7. Evaluation Protocol

Report all of the following:

- DER at collar 0 s and 0.25 s.
- Miss, false alarm, and speaker confusion.
- JER.
- Speaker-count accuracy.
- DER by number of speakers (2/3/4).
- DER by overlap condition (no overlap, 2-speaker overlap, 3-4 speaker overlap).
- Per-meeting worst-case analysis.

Use the same clustering hyperparameters across systems. Do not tune clustering on the test set.

---

## 8. Licensing and Compliance

- Model weights: CC BY-NC 4.0, non-commercial.
- Dataset: CC BY-SA 4.0.
- Fine-tuned weights are likely subject to the model's non-commercial restriction.
- The dataset's share-alike terms may affect redistribution of derivatives.
- Obtain legal review before any commercial deployment.
- If commercial use is required, choose a permissively licensed model and dataset, or train from scratch.

---

## 9. Compute and Storage

Current state:

- Repo size: ~203.8 GB.
- Free space on `D:`: ~33.8 GB.
- Parquet copy: ~122 GB.
- Extracted audio: ~71.6 GB.
- Simulated data: ~14.4 GB.

Actions:

- Delete the parquet copy after extraction to free ~122 GB.
- Reserve space for the DiariZen environment and checkpoints.
- Use 1-4 GPUs for the pilot; scale only after the pilot succeeds.
- Use `num_workers` 4-8 for data loading.
- Keep checkpoints and logs outside the git repository.

---

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| Overfitting to 20 speakers | Speaker-disjoint splits; pilot first; add real speakers |
| Catastrophic forgetting | Freeze WavLM initially; low LR; early stopping |
| Misleading dev scores | Remove speaker and source-cut leakage |
| Clean-to-far-field mismatch | Add reverb/noise or collect real far-field data |
| Pruned model capacity | Compare against unpruned model; consider prune-after-fine-tune |
| Embedding/PLDA mismatch | Adapt WeSpeaker/PLDA or evaluate clustering separately |
| License conflict | Legal review before training or deployment |
| Disk exhaustion | Delete parquet copy; monitor checkpoint size |

---

## 11. Open Questions

1. Is the target real-world Khmer meetings/calls, or only this simulated corpus?
2. Is there any real Khmer conversational audio with speaker labels? How many hours and speakers?
3. Is the intended use research-only or commercial?
4. What GPU model and count are available?
5. Should we fine-tune the pruned v2 checkpoint, or fine-tune an unpruned DiariZen model and prune afterward?

---

## 12. Immediate Next Actions

1. Answer the open questions above.
2. Collect or identify 1-2 h of real Khmer conversational audio for evaluation.
3. Set up DiariZen and run `scripts/run_oneshot.py --stages prepare --yes`.
4. Run the zero-shot v2 baseline with `scripts/run_oneshot.py --stages baseline score --yes`.
5. Start the 10-20 h frozen-WavLM pilot with `scripts/run_oneshot.py --stages train export score --yes`.
6. Only unfreeze WavLM (`--unfreeze-wavlm`) if the frozen pilot improves DER.

---

## Appendix A - Current Artifacts

| Artifact | Path |
|---|---|
| Simulated audio | `data/sim/<topic>/audio/*.wav` |
| Simulated RTTM | `data/sim/<topic>/rttm/*.rttm` |
| Lhotse train manifest | `data/sim/manifests/synth-khmer-train.jsonl.gz` |
| Lhotse dev manifest | `data/sim/manifests/synth-khmer-dev.jsonl.gz` |
| NeMo train manifest | `data/sim/manifests/nemo-train.json` |
| NeMo dev manifest | `data/sim/manifests/nemo-dev.json` |
| Validation report | `data/sim/manifests/validation.json` |
| Simulation config | `FastMSS/recipes/khmer.yaml` |
| Training-set builder | `build_training_set.py` |

## Appendix B - One-Shot Pilot Implementation

All pilot code lives in `scripts/` at the repository root.

| Artifact | Purpose |
|---|---|
| `scripts/prepare_data.py` | Builds validated, speaker-disjoint `wav.scp` / `rttm/` / `rttm.txt` / `all.uem` bundles and a `split.json` leakage report |
| `scripts/download_model.py` | Downloads the v2 model and WeSpeaker embedding model |
| `scripts/run_zero_shot.py` | Runs `DiariZenPipeline` on a `wav.scp` and writes RTTM files |
| `configs/finetune_khmer.toml` | Training config template, rendered with absolute paths |
| `scripts/check_checkpoint.py` | Verifies that the HF inference checkpoint keys match the v2 training model |
| `scripts/run_oneshot.py` | Orchestrates download, check, prepare, render, baseline, train, export, score, and the gate summary |
| `scripts/export_model.py` | Copies a training checkpoint over the base inference model |
| `scripts/score_rttm.py` | Wraps `DiariZen/dscore/score.py` and parses overall DER into `metrics.json` |

Gate: the pilot passes when finetuned DER is at least 1.0 absolute point lower
than the zero-shot baseline on the same test split. This only justifies building
a real Khmer evaluation set and a larger pilot; it does not justify release.

## Appendix C - Model and Recipe References

- Model: `https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2`
- Code: `https://github.com/BUTSpeechFIT/DiariZen`
- Training recipe: `recipes/diar_ssl/run_stage.sh`
- Training config: `recipes/diar_ssl/conf/wavlm_updated_conformer.toml`
- Dataset: `https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset`



