# DiariZen Khmer Pilot

Gated one-shot pilot for fine-tuning `BUT-FIT/diarizen-wavlm-large-s80-md-v2`
on simulated Khmer meetings produced by FastMSS.

This is not a production recipe. It answers one question: does a small,
speaker-disjoint Khmer pilot improve on the zero-shot baseline? The full audit
and phased plan are in `docs/KHMER_DIARIZATION_FINETUNE_PLAN.md`. See
`docs/STATUS_AND_NEXT.md` for completed work and the exact next commands, and
`docs/PLAN.md` for the roadmap, including real-meeting solo playback.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── .github/workflows/ci.yml
├── configs/
│   ├── finetune_khmer.toml
│   └── finetune_khmer_8epoch.toml
├── docs/
│   ├── KHMER_DIARIZATION_FINETUNE_PLAN.md
│   ├── PLAN.md
│   └── TODOS.md
├── scripts/
│   ├── run_oneshot.py
│   ├── prepare_data.py
│   ├── download_model.py
│   ├── check_checkpoint.py
│   ├── run_zero_shot.py
│   ├── export_model.py
│   ├── score_rttm.py
│   ├── monitor_training.py
│   ├── build_report.py
│   ├── build_finetune_report.py
│   ├── build_dev_demo.py
│   ├── build_solo_audio.py
│   ├── extract_target.py
│   └── split_meeting.py
├── artifacts/dev_demo/       # playable 10-meeting demo (committed)
├── report.html               # frozen baseline report
├── report_finetuned.html     # baseline-vs-finetuned report
├── data/                     # generated bundles (ignored)
├── models/                   # downloaded checkpoints (ignored)
├── runs/                     # pilot outputs (ignored)
├── runs_full/                # full-run outputs (ignored)
└── DiariZen/                 # external framework clone (ignored)
```

Generated data, model weights, checkpoints, and run outputs are deliberately not
committed. `artifacts/dev_demo/` is the only generated artifact included because
it is the self-contained demo.

## Setup

Clone DiariZen into the repository root:

```
git clone https://github.com/BUTSpeechFIT/DiariZen.git
```

Install DiariZen following its README, including the `pyannote-audio` editable
install, `torch`, and the `dscore` submodule. Then install the pilot extras:

```
pip install -r requirements.txt
```

The training stage launches `DiariZen/recipes/diar_ssl/run_dual_opt.py`, so it
needs a working PyTorch install and enough VRAM for 16 s WavLM-Large chunks.
The pilot defaults to a frozen-WavLM config and a small batch size.

The data preparation stage reads the FastMSS sibling checkout by default. Set
`--fastmss-root` if it lives elsewhere:

```
python scripts/run_oneshot.py --fastmss-root /path/to/FastMSS --dry-run
```

This environment is Windows with a 4 GB RTX 3050. The scripts add these
compatibility workarounds:

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted local HF checkpoints.
- `soundfile` fallback for `torchaudio` WAV loading when torchcodec/FFmpeg is
  unavailable.
- Perl is required for dscore's `md-eval-22.pl`; install it with
  `conda install -n fastmss -c conda-forge perl`.
- The embedding directory is named `embedding-resnet34` on purpose: pyannote
  selects its ONNX WeSpeaker loader when the path contains `wespeaker`, but the
  HF repo only ships a PyTorch checkpoint.
- `--infer-batch-size` defaults to 2 for the 4 GB GPU.

## Results

The 2-file smoke test scored DER 3.39%, but it was optimistic. The full 82-file
zero-shot baseline on the 1.699 h test split scored **DER 9.58%** at collar
0.25 (JER 19.95%, B3-F1 0.89). The gate therefore required fine-tuned DER
<= 8.58%.

The full 8-epoch frozen-WavLM run improved the same split to **DER 4.62%**
(JER 15.53%, B3-F1 0.96), a **4.96-point** gain and a gate PASS. The best
checkpoint was epoch 8 by validation NLL loss (0.1506), with validation DER
3.10%. See `runs_full/scores/finetuned_holdout/metrics.json`.

The earlier 1.03 h pilot remains in `runs/`; the full run is in `runs_full/`.
Neither directory is committed.

## Reports

`report.html` is a self-contained visual report of the baseline process,
metrics, and per-file DER table. It is preserved unchanged (SHA-256
`B3847EECEC35536CD0ED1955C96A95C01584B076AC965BE29E51DE5AEA78FB6D`). Some
links point to `runs/` outputs that are intentionally not committed. Rebuild it
after rescoring the baseline with:

```
python scripts/build_report.py
```

`report_finetuned.html` compares the baseline and fine-tuned results, including
per-file deltas and the gate decision. Its links to `runs_full/` are also local
by design. Rebuild it with:

```
python scripts/build_finetune_report.py
```

`artifacts/dev_demo/index.html` is a playable 10-meeting dev artifact with audio,
waveforms, interactive timelines, fine-tuned/baseline/reference tabs, and exact
per-speaker solo playback. On those 10 files the fine-tuned model scores 4.24%
DER versus 5.68% for the baseline. Rebuild the page after rerunning demo
inference and scoring with:

```
python scripts/build_dev_demo.py
```

## Speaker isolation

The synthetic demo has exact source stems because FastMSS simulated each meeting
with `reverberate: false`, `use_fir: false`, and `add_noise: false`. Rebuild the
stems and then the demo page with:

```
python scripts/build_solo_audio.py
python scripts/build_dev_demo.py
```

The page exposes a reference-speaker selector and a **Play solo** / **Full mix**
toggle. Each solo WAV is the target speaker only; the other speakers are removed
exactly, not by a separator. Reconstruction SNR is 63-78 dB against the mixture,
with residual at the 16-bit PCM floor.

For arbitrary real audio, exact removal is impossible from a single mixed track.
`extract_target.py` provides an approximate model-based path using SpeechBrain
SepFormer plus ECAPA enrollment scoring:

```
python scripts/extract_target.py \
  --mix meeting.wav \
  --enroll target_speaker.wav \
  --out target_only.wav \
  --rttm meeting.rttm --speaker m-adt1-0003 \
  --num-speakers 3
```

Without `--rttm`, it processes all chunks and keeps only chunks whose separated
source is close enough to the enrollment. With `--rttm`, it only processes
target-active regions and level-matches the output to the mixture. The 2-source
WHAMR model is used for `--num-speakers 2`; the Libri3Mix model is used for 3+.
The SpeechBrain checkpoints are downloaded on first use. Separation is
approximate and degrades with overlap, noise, more than three active speakers,
and domain mismatch. On two demo meetings, SDR against the exact stem was about
6.3 dB for a clean 2-speaker case and 3.9-4.3 dB for 3-speaker targets. This is
a demo path, not a production separator.

To split every speaker in a real meeting, run diarization first to get an RTTM,
then run the batch wrapper:

```
python scripts/split_meeting.py \
  --mix meeting.wav \
  --rttm meeting.rttm \
  --out-dir stems/meeting \
  --num-speakers 3
```

It auto-picks a target-only enrollment interval for each RTTM speaker, calls
`extract_target.py`, and writes `stems/meeting/<speaker>.wav` plus
`manifest.json`. If a speaker never has a non-overlapping region, put a clean
clip at `<enroll-dir>/<speaker>.wav` and pass `--enroll-dir`; otherwise the
auto-selected enrollment contains other speakers and quality drops. Diarization
cuts alone do not remove overlapping voices; this wrapper uses separation to
approximate that.

## Quick start

Always dry-run first:

```
python scripts/run_oneshot.py --dry-run
```

Prepare the holdout bundle and inspect the split report:

```
python scripts/run_oneshot.py --stages prepare --yes
```

Render and run the full gated pilot. On the 4 GB GPU, use batch 1 with gradient
accumulation 4 and a small validation batch:

```
python scripts/run_oneshot.py --stages all --yes --train-batch-size 1 --val-batch-size 2 --grad-accum 4 --num-workers 0
```

The probe and pilot used this safe setting; training takes about 40 s per epoch
and validation about 4-5 min per epoch on the 11.5 h dev split.

Unfreeze WavLM only after a frozen-WavLM run shows promise:

```
python scripts/run_oneshot.py --stages all --yes --unfreeze-wavlm
```

Cross-validation instead of a single holdout:

```
python scripts/run_oneshot.py --stages prepare render train export score --yes --mode cv --folds 5
```

Score only what already exists:

```
python scripts/run_oneshot.py --stages score --yes
```

## Stages

- `download`: downloads the model to `models/diarizen-wavlm-large-s80-md-v2` and the embedding to `models/embedding-resnet34`.
- `check`: instantiates the v2 model and reports how many checkpoint keys match; abort if under 90%.
- `prepare`: reads the FastMSS Lhotse manifests, validates audio/RTTM, and writes speaker-disjoint `holdout/` or `fold_i/` bundles under `data/diarizen/khmer`.
- `render`: writes `runs/configs/khmer_finetune_<bundle>.toml` from the template in `configs/`.
- `baseline`: runs zero-shot inference on each test split into `runs/zero_shot/<bundle>`.
- `train`: runs `accelerate launch run_dual_opt.py -C <config> -M train` from `DiariZen/recipes/diar_ssl`.
- `export`: copies the best checkpoint over the base inference model into `runs/exported/khmer_finetune_<bundle>`, then runs fine-tuned inference into `runs/finetuned/khmer_finetune_<bundle>`.
- `score`: scores baseline and fine-tuned RTTMs with dscore and prints the gate summary.

## Outputs

```
data/diarizen/khmer/<bundle>/{train,dev,test}/{wav.scp,rttm,rttm.txt,all.uem,split.json}
runs/configs/khmer_finetune_<bundle>.toml
runs/exp/khmer_finetune_<bundle>/checkpoints/{best,epoch_XXXX}/pytorch_model.bin
runs/exported/khmer_finetune_<bundle>/
runs/zero_shot/<bundle>/
runs/finetuned/khmer_finetune_<bundle>/
runs/scores/{baseline,finetuned}_<bundle>/{score.txt,metrics.json}
artifacts/dev_demo/{audio,ref,pred_*,score_*,waveforms,solo}/
artifacts/dev_demo/solo_manifest.json
report.html
report_finetuned.html
```

The full run used `runs_full/` instead of `runs/`. The 10-meeting demo and its
exact solo stems are in `artifacts/dev_demo/`; model-based extraction outputs
from the checks are under `artifacts/dev_demo/model_extract/` and are ignored by
Git.

## Gate

`runs_full/scores/*/metrics.json` holds the overall DER parsed from dscore. A
pilot passes when fine-tuned DER is at least 1.0 point lower than the zero-shot
baseline on the same test split. Anything smaller is inside the noise expected
from a small 82-file holdout. The full 8-epoch run passed:
9.58% -> 4.62%, delta 4.96 points.

This gate is deliberately weak. It only justifies spending time on a real Khmer
conversational evaluation set, better speaker coverage, and a larger training
mixture. It does not justify shipping a model.

## Caveats

- The full training bundle has 16 speakers (17 in dev, 16 in test) and a small
  number of speakers dominate the source cuts. Domain transfer to real Khmer
  meetings is likely poor.
- The model is CC BY-NC 4.0 and the source corpus is CC BY-SA 4.0. Treat this
  pilot as non-commercial and re-check licensing before any release.
- `prepare_data.py` never copies audio. `wav.scp` points at the existing 16 kHz
  WAVs in `FastMSS/data/sim`.
- Training consumes the whole audio directly, so keep the pilot at 10-20 h
  until the gate and a real evaluation set exist.
