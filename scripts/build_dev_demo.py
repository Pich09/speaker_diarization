import html
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "dev_demo"
AUDIO_DIR = DEMO / "audio"
REF_DIR = DEMO / "ref"
FT_DIR = DEMO / "pred_finetuned"
BASE_DIR = DEMO / "pred_baseline"
FT_SCORE_DIR = DEMO / "score_finetuned"
BASE_SCORE_DIR = DEMO / "score_baseline"
WAVE_DIR = DEMO / "waveforms"
INPUT_SCP = DEMO / "input_all.scp"
SOLO_MANIFEST = DEMO / "solo_manifest.json"
OUT = DEMO / "index.html"
DEFAULT_ID = "00716422-f4a1-4a37-8170-d0eb2e110103"

PALETTE = ["#6ea8ff", "#3ddc97", "#ffbf69", "#ff6b81", "#8b7cff", "#4cc9f0"]


def parse_rttm(path):
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        raw = parts[-2] if parts[-2] != "<NA>" else parts[-3]
        start = float(parts[3])
        end = start + float(parts[4])
        segments.append({"start": start, "end": end, "raw": raw})
    segments.sort(key=lambda item: (item["start"], item["end"]))
    return segments


def assign_speakers(segments):
    order = []
    for segment in segments:
        if segment["raw"] not in order:
            order.append(segment["raw"])
    mapping = {raw: f"S{index + 1}" for index, raw in enumerate(order)}
    colors = {raw: PALETTE[index % len(PALETTE)] for index, raw in enumerate(order)}
    for segment in segments:
        segment["speaker"] = mapping[segment["raw"]]
        segment["color"] = colors[segment["raw"]]
    speakers = [
        {"id": mapping[raw], "raw": raw, "color": colors[raw]}
        for raw in order
    ]
    return speakers


def compute_stats(segments, duration):
    events = []
    for segment in segments:
        events.append((segment["start"], 1))
        events.append((segment["end"], -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    previous = None
    union = 0.0
    overlap = 0.0
    for time, delta in events:
        if previous is not None and time > previous:
            if active >= 1:
                union += time - previous
            if active >= 2:
                overlap += time - previous
        active += delta
        previous = time
    speech = sum(segment["end"] - segment["start"] for segment in segments)
    return {
        "speakers": len({segment["speaker"] for segment in segments}),
        "turns": len(segments),
        "speech": speech,
        "union": union,
        "overlap": overlap,
        "overlap_pct": (overlap / duration * 100.0) if duration else 0.0,
    }


def load_scores(score_dir):
    metrics = json.loads((score_dir / "metrics.json").read_text(encoding="utf-8"))
    overall = {
        "der": metrics["der"],
        "jer": None,
        "b3_f1": None,
        "files": metrics.get("num_scored", metrics.get("num_ref")),
    }
    files = {}
    for line in (score_dir / "score.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] == "File" or parts[0].startswith("-"):
            continue
        if line.startswith("*** OVERALL ***"):
            fields = parts[3:]
            overall["der"] = float(fields[0])
            overall["jer"] = float(fields[1])
            overall["b3_f1"] = float(fields[4])
            continue
        if re.fullmatch(r"[0-9a-fA-F-]{36}", parts[0]):
            fields = parts[1:]
            files[parts[0]] = {
                "der": float(fields[0]),
                "jer": float(fields[1]),
                "b3_f1": float(fields[4]),
            }
    return overall, files


def make_waveform(audio_path, out_path):
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    bins = 1400
    edges = np.linspace(0, len(mono), bins + 1, dtype=int)
    envelope = np.array(
        [np.max(np.abs(mono[start:end])) if end > start else 0.0 for start, end in zip(edges[:-1], edges[1:])]
    )
    peak = float(envelope.max()) or 1.0
    envelope = envelope / peak
    figure = Figure(figsize=(12, 1.5), dpi=160)
    figure.patch.set_alpha(0)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_facecolor("none")
    x = np.linspace(0, 1, len(envelope))
    axes.fill_between(x, envelope, -envelope, color="#6ea8ff", linewidth=0, alpha=0.9)
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 1)
    axes.axis("off")
    figure.savefig(out_path, transparent=True, dpi=160, pad_inches=0)
    return sample_rate


def build_system(path, label, description, score, duration):
    segments = parse_rttm(path)
    speakers = assign_speakers(segments)
    return {
        "label": label,
        "description": description,
        "rttm": path.relative_to(DEMO).as_posix(),
        "segments": segments,
        "speakers": speakers,
        "stats": compute_stats(segments, duration),
        "score": score,
    }


def fmt_time(seconds):
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}"


def load_meeting_ids():
    ids = []
    for line in INPUT_SCP.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(line.split()[0])
    return ids


def main():
    meeting_ids = load_meeting_ids()
    overall_ft, ft_files = load_scores(FT_SCORE_DIR)
    overall_base, base_files = load_scores(BASE_SCORE_DIR)
    solo_data = {}
    if SOLO_MANIFEST.exists():
        solo_data = json.loads(SOLO_MANIFEST.read_text(encoding="utf-8")).get("meetings", {})

    meetings = []
    total_duration = 0.0
    for meeting_id in meeting_ids:
        audio_path = AUDIO_DIR / f"{meeting_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        info = sf.info(audio_path)
        duration = float(info.duration)
        total_duration += duration
        wave_path = WAVE_DIR / f"{meeting_id}.png"
        make_waveform(audio_path, wave_path)

        ft_score = ft_files.get(meeting_id)
        base_score = base_files.get(meeting_id)
        if ft_score is None or base_score is None:
            raise KeyError(f"Missing score row for {meeting_id}")

        systems = {
            "finetuned": build_system(
                FT_DIR / f"{meeting_id}.rttm",
                "Fine-tuned",
                "Frozen-WavLM fine-tune; best checkpoint at epoch 8 on the full bundle.",
                ft_score,
                duration,
            ),
            "baseline": build_system(
                BASE_DIR / f"{meeting_id}.rttm",
                "Zero-shot baseline",
                "Original v2 model without Khmer fine-tuning.",
                base_score,
                duration,
            ),
            "reference": build_system(
                REF_DIR / f"{meeting_id}.rttm",
                "Reference",
                "Ground-truth RTTM from the dev split.",
                None,
                duration,
            ),
        }
        solo_meeting = solo_data.get(meeting_id, {})
        reference_by_raw = {speaker["raw"]: speaker for speaker in systems["reference"]["speakers"]}
        solo_speakers = []
        for solo_speaker in solo_meeting.get("speakers", []):
            reference = reference_by_raw.get(solo_speaker["raw"])
            if reference is None:
                continue
            solo_speakers.append(
                {
                    "id": reference["id"],
                    "raw": reference["raw"],
                    "color": reference["color"],
                    "audio": solo_speaker["audio"],
                    "speech": solo_speaker["speech"],
                    "turns": solo_speaker["turns"],
                    "rmsDb": solo_speaker["rms_db"],
                }
            )
        meetings.append(
            {
                "id": meeting_id,
                "duration": duration,
                "durationFmt": fmt_time(duration),
                "audio": audio_path.relative_to(DEMO).as_posix(),
                "waveform": wave_path.relative_to(DEMO).as_posix(),
                "ftScoreLink": (FT_SCORE_DIR / "score.txt").relative_to(DEMO).as_posix(),
                "baseScoreLink": (BASE_SCORE_DIR / "score.txt").relative_to(DEMO).as_posix(),
                "solo": solo_speakers,
                "soloVerification": solo_meeting.get("verification"),
                "systems": systems,
            }
        )

    if not meetings:
        raise RuntimeError("No demo meetings found")

    demo = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overall": {
            "finetuned": overall_ft,
            "baseline": overall_base,
            "delta": overall_base["der"] - overall_ft["der"],
            "meeting_count": len(meetings),
            "total_duration": total_duration,
        },
        "meetings": meetings,
    }
    default_id = DEFAULT_ID if any(item["id"] == DEFAULT_ID for item in meetings) else meetings[0]["id"]

    replacements = {
        "__GENERATED__": demo["generated"],
        "__MEETING_COUNT__": str(len(meetings)),
        "__TOTAL_DURATION__": f"{total_duration / 60:.1f} min",
        "__OVERALL_FT_DER__": f"{overall_ft['der']:.2f}",
        "__OVERALL_FT_JER__": f"{overall_ft['jer']:.2f}",
        "__OVERALL_FT_B3__": f"{overall_ft['b3_f1']:.3f}",
        "__OVERALL_BASE_DER__": f"{overall_base['der']:.2f}",
        "__OVERALL_BASE_JER__": f"{overall_base['jer']:.2f}",
        "__OVERALL_BASE_B3__": f"{overall_base['b3_f1']:.3f}",
        "__OVERALL_DELTA__": f"{overall_base['der'] - overall_ft['der']:.2f}",
        "__DEFAULT_ID_JSON__": json.dumps(default_id),
        "__DEMO_JSON__": json.dumps(demo, ensure_ascii=False),
    }
    page = TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        f"meetings={len(meetings)} total={total_duration:.1f}s "
        f"fine={overall_ft['der']:.2f} baseline={overall_base['der']:.2f} "
        f"delta={overall_base['der'] - overall_ft['der']:.2f}"
    )


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Khmer Diarization Demo - 10 Dev Meetings</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #121a31;
    --panel-2: #17213d;
    --line: #2a3658;
    --text: #e8edf8;
    --muted: #9aa8c7;
    --accent: #6ea8ff;
    --good: #3ddc97;
    --warn: #ffbf69;
    --bad: #ff6b81;
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--text);
    background:
      radial-gradient(900px 500px at 8% -10%, rgba(110,168,255,.20), transparent 60%),
      radial-gradient(800px 500px at 100% 0%, rgba(139,124,255,.16), transparent 55%),
      var(--bg);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    line-height: 1.55;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .wrap { width: min(1180px, calc(100% - 40px)); margin: 0 auto; }
  header { padding: 52px 0 24px; }
  .eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); box-shadow: 0 0 0 4px rgba(61,220,151,.14); }
  h1 { margin: 12px 0 10px; font-size: clamp(32px, 5vw, 56px); line-height: 1.05; letter-spacing: -0.03em; }
  h1 span { color: var(--accent); }
  .lede { max-width: 920px; color: var(--muted); font-size: 17px; margin: 0; }
  .lede code { color: var(--text); }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
  .badge { border: 1px solid var(--line); background: rgba(20,29,54,.75); color: var(--muted); padding: 6px 10px; border-radius: 999px; font-size: 12px; }
  .badge.good { color: var(--good); border-color: rgba(61,220,151,.35); background: rgba(61,220,151,.08); }
  main { padding-bottom: 56px; }
  section { margin: 24px 0; }
  .section-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
  h2 { margin: 0; font-size: 22px; letter-spacing: -0.02em; }
  .section-head p { margin: 0; color: var(--muted); font-size: 13px; }
  .cards { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
  .card, .panel { border: 1px solid var(--line); background: linear-gradient(180deg, rgba(23,33,61,.92), rgba(18,26,49,.92)); border-radius: 16px; box-shadow: var(--shadow); }
  .card { padding: 16px; min-height: 116px; display: flex; flex-direction: column; justify-content: space-between; }
  .card.primary { border-color: rgba(110,168,255,.55); background: linear-gradient(180deg, rgba(110,168,255,.18), rgba(18,26,49,.92)); }
  .card.good-card { border-color: rgba(61,220,151,.35); }
  .card-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
  .card-value { font-size: 26px; font-weight: 800; letter-spacing: -0.03em; margin: 8px 0 2px; }
  .card-note { color: var(--muted); font-size: 12px; }
  .good-text { color: var(--good); }
  .bad-text { color: var(--bad); }
  .panel { padding: 20px; }
  .panel h3 { margin: 0 0 12px; font-size: 16px; }
  .picker-panel { padding: 16px 18px; }
  .picker-row { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
  .picker-row label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
  select {
    flex: 1 1 460px;
    min-width: 280px;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #0f1730;
    color: var(--text);
    padding: 10px 12px;
    font-size: 13px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  }
  button.step {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(11,16,32,.55);
    color: var(--text);
    padding: 10px 13px;
    cursor: pointer;
    font-size: 13px;
  }
  button.step:hover { border-color: var(--accent); }
  #meetingLabel { flex-basis: 100%; color: var(--muted); font-size: 12px; }
  .audio-panel { padding: 18px 20px 16px; }
  .audio-head { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 10px; }
  .audio-head strong { font-size: 16px; }
  #clock { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; }
  audio { width: 100%; margin: 6px 0 14px; }
  .solo-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 0 0 12px; padding: 10px 12px; border: 1px solid rgba(61,220,151,.3); border-radius: 12px; background: rgba(61,220,151,.06); }
  .solo-row label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
  .solo-select { flex: 1 1 260px; min-width: 180px; border: 1px solid var(--line); border-radius: 10px; background: #0f1730; color: var(--text); padding: 9px 11px; font-size: 13px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
  .solo-status { flex-basis: 100%; color: var(--muted); font-size: 12px; }
  button.step.active { border-color: var(--good); color: var(--good); background: rgba(61,220,151,.12); }
  .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 12px; }
  .tab { border: 1px solid var(--line); background: rgba(11,16,32,.55); color: var(--muted); border-radius: 999px; padding: 8px 13px; font-size: 13px; cursor: pointer; }
  .tab:hover { color: var(--text); border-color: rgba(110,168,255,.5); }
  .tab.active { color: var(--text); border-color: var(--accent); background: rgba(110,168,255,.14); }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; min-height: 26px; }
  .chip { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); background: rgba(11,16,32,.35); border-radius: 999px; padding: 4px 9px; font-size: 12px; color: var(--muted); }
  .chip i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .timeline-body { position: relative; border: 1px solid var(--line); border-radius: 14px; background: rgba(5,9,20,.35); padding: 10px 12px 12px; }
  .ruler { position: relative; height: 22px; margin-left: 100px; color: var(--muted); font-size: 11px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
  .tick { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
  .tick::after { content: ""; position: absolute; left: 50%; top: 16px; width: 1px; height: 5px; background: var(--line); }
  .wave-wrap { position: relative; margin-left: 100px; height: 80px; border-radius: 8px; background: rgba(110,168,255,.05); overflow: hidden; }
  .waveform { width: 100%; height: 100%; object-fit: fill; display: block; opacity: .95; }
  .wave-playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent); left: 0; box-shadow: 0 0 8px rgba(110,168,255,.8); pointer-events: none; }
  .lanes { position: relative; margin-top: 10px; }
  .lane { display: flex; align-items: center; min-height: 42px; border-top: 1px solid rgba(42,54,88,.55); }
  .lane:first-child { border-top: 0; }
  .lane-label { flex: 0 0 100px; width: 100px; color: var(--muted); font-size: 12px; display: flex; align-items: center; gap: 7px; }
  .lane-label i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .lane-track { position: relative; flex: 1; height: 26px; border-radius: 7px; background: rgba(23,33,61,.85); overflow: hidden; cursor: crosshair; }
  .segment { position: absolute; top: 0; height: 100%; border-radius: 6px; opacity: .78; border: 1px solid rgba(255,255,255,.14); cursor: pointer; transition: opacity .12s ease, transform .12s ease; }
  .segment:hover { opacity: 1; transform: scaleY(1.08); }
  .segment.active { opacity: 1; box-shadow: 0 0 0 2px rgba(255,255,255,.75) inset; }
  .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent); box-shadow: 0 0 8px rgba(110,168,255,.8); pointer-events: none; }
  .hint { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
  .grid-2 { display: grid; grid-template-columns: .8fr 1.2fr; gap: 16px; }
  .stat-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .stat { border: 1px solid var(--line); background: rgba(11,16,32,.35); border-radius: 12px; padding: 11px 12px; }
  .stat span { display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .07em; }
  .stat strong { display: block; font-size: 20px; margin-top: 3px; }
  .system-title { font-size: 18px; font-weight: 800; letter-spacing: -0.02em; }
  .system-desc { margin: 4px 0 10px; color: var(--muted); font-size: 13px; }
  .system-score { margin: 0 0 12px; color: var(--muted); font-size: 13px; }
  .system-score strong { color: var(--text); }
  .link-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .link-row a { border: 1px solid var(--line); border-radius: 8px; padding: 6px 9px; font-size: 12px; background: rgba(11,16,32,.35); }
  .table-wrap { max-height: 340px; overflow: auto; border: 1px solid var(--line); border-radius: 12px; }
  .meetings-table { max-height: none; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
  th { position: sticky; top: 0; z-index: 2; color: var(--muted); background: #121a31; font-size: 11px; text-transform: uppercase; letter-spacing: .07em; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: rgba(110,168,255,.07); }
  tbody tr.selected { background: rgba(110,168,255,.16); }
  .mono { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: 12px; }
  .muted { color: var(--muted); }
  .small { font-size: 12px; }
  .artifact-list { display: grid; gap: 8px; }
  .artifact { padding: 9px 11px; border: 1px solid var(--line); border-radius: 10px; background: rgba(11,16,32,.3); }
  .artifact code { color: var(--text); font-size: 12px; word-break: break-all; }
  .artifact span { color: var(--muted); font-size: 12px; }
  .callout { border-left: 3px solid var(--accent); background: rgba(110,168,255,.08); border-radius: 0 12px 12px 0; padding: 12px 14px; color: var(--muted); font-size: 13px; }
  .callout strong { color: var(--text); }
  ul.clean { margin: 0; padding-left: 20px; color: var(--muted); font-size: 13px; }
  ul.clean li { margin: 6px 0; }
  footer { border-top: 1px solid var(--line); padding: 20px 0 38px; color: var(--muted); font-size: 12px; }
  @media (max-width: 1100px) { .cards { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 980px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid-2 { grid-template-columns: 1fr; } }
  @media (max-width: 620px) { .wrap { width: min(100% - 24px, 1180px); } .cards { grid-template-columns: 1fr; } .ruler, .wave-wrap { margin-left: 74px; } .lane-label { flex-basis: 74px; width: 74px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow"><span class="dot"></span> Dev audio demo &middot; 10 meetings</div>
    <h1>Khmer <span>diarization</span> demo</h1>
    <p class="lede">
      Ten dev meetings with playable audio, fine-tuned diarization, zero-shot baseline, and reference RTTM.
      Pick a meeting, switch systems, click the timeline or a segment to seek, and isolate any reference speaker
      with an exact source stem.
    </p>
    <div class="badges">
      <span class="badge good" id="badgeFine">Fine-tuned __OVERALL_FT_DER__% DER</span>
      <span class="badge" id="badgeBase">Baseline __OVERALL_BASE_DER__% DER</span>
      <span class="badge" id="badgeSpeakers">3 speakers</span>
      <span class="badge" id="badgeFiles">10 dev meetings</span>
      <span class="badge">Collar 0.25 s</span>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>10-file dev subset</h2>
        <p>Overall results on the selected 10 dev meetings; the 82-file test result is separate</p>
      </div>
      <div class="cards">
        <article class="card primary">
          <div class="card-label">Fine-tuned DER</div>
          <div class="card-value">__OVERALL_FT_DER__%</div>
          <div class="card-note">JER __OVERALL_FT_JER__% &middot; B3-F1 __OVERALL_FT_B3__</div>
        </article>
        <article class="card">
          <div class="card-label">Baseline DER</div>
          <div class="card-value">__OVERALL_BASE_DER__%</div>
          <div class="card-note">JER __OVERALL_BASE_JER__% &middot; B3-F1 __OVERALL_BASE_B3__</div>
        </article>
        <article class="card good-card">
          <div class="card-label">Improvement</div>
          <div class="card-value good-text">__OVERALL_DELTA__ pt</div>
          <div class="card-note">fine-tuned minus baseline</div>
        </article>
        <article class="card">
          <div class="card-label">Meetings</div>
          <div class="card-value">__MEETING_COUNT__</div>
          <div class="card-note">__TOTAL_DURATION__ of audio</div>
        </article>
        <article class="card">
          <div class="card-label">Total duration</div>
          <div class="card-value">__TOTAL_DURATION__</div>
          <div class="card-note">__MEETING_COUNT__ meetings</div>
        </article>
        <article class="card">
          <div class="card-label">Collar</div>
          <div class="card-value">0.25s</div>
          <div class="card-note">DER scoring</div>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Select a meeting</h2>
        <p>All audio and RTTMs are local files in this artifact folder</p>
      </div>
      <div class="panel picker-panel">
        <div class="picker-row">
          <label for="meetingSelect">Meeting</label>
          <select id="meetingSelect"></select>
          <button class="step" id="prevMeeting">Previous</button>
          <button class="step" id="nextMeeting">Next</button>
          <span id="meetingLabel"></span>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>File at a glance</h2>
        <p id="fileGlanceNote">Per-file DER on the selected dev meeting</p>
      </div>
      <div class="cards">
        <article class="card primary">
          <div class="card-label">Fine-tuned DER</div>
          <div class="card-value" id="cardFineDer">--</div>
          <div class="card-note" id="cardFineNote">collar 0.25 s</div>
        </article>
        <article class="card">
          <div class="card-label">Baseline DER</div>
          <div class="card-value" id="cardBaseDer">--</div>
          <div class="card-note" id="cardBaseNote">collar 0.25 s</div>
        </article>
        <article class="card good-card">
          <div class="card-label">Improvement</div>
          <div class="card-value good-text" id="cardDelta">--</div>
          <div class="card-note">fine-tuned minus baseline</div>
        </article>
        <article class="card">
          <div class="card-label">Reference speakers</div>
          <div class="card-value" id="cardSpeakers">--</div>
          <div class="card-note" id="cardTurns">-- turns</div>
        </article>
        <article class="card">
          <div class="card-label">Reference speech</div>
          <div class="card-value" id="cardSpeech">--</div>
          <div class="card-note">sum of turn durations</div>
        </article>
        <article class="card">
          <div class="card-label">Reference overlap</div>
          <div class="card-value" id="cardOverlap2">--</div>
          <div class="card-note" id="cardOverlapPct">--</div>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Audio and timeline</h2>
        <p>Click a lane to seek; click a segment or table row to jump</p>
      </div>
      <div class="panel audio-panel">
        <div class="audio-head">
          <strong id="audioTitle">Playable meeting audio</strong>
          <span id="clock">00:00.0 / 00:00.0</span>
        </div>
        <audio id="audio" controls preload="metadata"></audio>
        <div class="solo-row">
          <label for="soloSpeaker">Isolate speaker</label>
          <select id="soloSpeaker" class="solo-select"></select>
          <button class="step" id="soloToggle" type="button">Play solo</button>
          <span class="solo-status" id="soloStatus"></span>
        </div>
        <div class="tabs" id="tabs">
          <button class="tab" data-system="finetuned">Fine-tuned</button>
          <button class="tab" data-system="baseline">Zero-shot baseline</button>
          <button class="tab" data-system="reference">Reference</button>
        </div>
        <div class="legend" id="legend"></div>
        <div class="timeline-body">
          <div class="ruler" id="ruler"></div>
          <div class="wave-wrap">
            <img class="waveform" id="waveform" alt="Waveform">
            <div class="wave-playhead" id="wavePlayhead"></div>
          </div>
          <div class="lanes" id="lanes"></div>
        </div>
        <p class="hint">Timestamps are in seconds; the playhead follows the audio. Segments in the same time range on different lanes indicate overlap.</p>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>Selected system</h3>
        <div id="systemMeta"></div>
        <div class="stat-grid" id="systemStats"></div>
        <div class="link-row" id="systemLinks"></div>
      </div>
      <div class="panel">
        <h3>Segments</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Start</th><th>End</th><th>Duration</th><th>Speaker</th><th>Raw label</th></tr>
            </thead>
            <tbody id="segments"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>All dev meetings</h2>
        <p>Click a row to switch the player and timeline</p>
      </div>
      <div class="panel">
        <div class="table-wrap meetings-table">
          <table>
            <thead>
              <tr><th>Meeting</th><th>Duration</th><th>Speakers</th><th>Fine-tuned DER</th><th>Baseline DER</th><th>Delta</th></tr>
            </thead>
            <tbody id="meetingTable"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>Artifacts</h3>
        <div class="artifact-list" id="artifactList"></div>
      </div>
      <div class="panel">
        <h3>Reading the demo</h3>
        <div class="callout" id="callout"></div>
        <h3 style="margin-top:18px">Caveats</h3>
        <ul class="clean">
          <li>These are 10 dev meetings, not the 82-file test set; per-file DER is noisy.</li>
          <li>The audio is simulated clean read speech, not real Khmer conversation.</li>
          <li>Only the segmentation model was fine-tuned; embedding and PLDA are unchanged.</li>
          <li>Speaker labels in predictions are clusters, not real identities; the mapping is arbitrary.</li>
          <li>Exact solo stems work only for these synthetic meetings, where the simulator kept the source cuts separate; arbitrary real audio needs a separator and may leak artifacts.</li>
        </ul>
      </div>
    </section>
  </main>

  <footer>
    Generated __GENERATED__ &middot; DiariZen Khmer dev demo &middot; 10 meetings, __TOTAL_DURATION__ &middot; audio, waveforms, and RTTMs are local artifact files.
  </footer>
</div>

<script>
  const DEMO = __DEMO_JSON__;
  const MEETINGS = DEMO.meetings;
  const BY_ID = Object.fromEntries(MEETINGS.map((item) => [item.id, item]));
  const DEFAULT_ID = __DEFAULT_ID_JSON__;
  const ORDER = ["finetuned", "baseline", "reference"];
  const audio = document.getElementById("audio");
  const soloSelect = document.getElementById("soloSpeaker");
  const soloToggle = document.getElementById("soloToggle");
  const soloStatus = document.getElementById("soloStatus");
  const lanes = document.getElementById("lanes");
  const playhead = document.createElement("div");
  playhead.className = "playhead";
  playhead.id = "playhead";
  let meeting = BY_ID[DEFAULT_ID] || MEETINGS[0];
  let duration = meeting.duration;
  let current = "finetuned";
  let soloSpeaker = null;
  let soloOn = false;

  function fmt(seconds) {
    const minutes = Math.floor(seconds / 60);
    const rest = (seconds % 60).toFixed(1).padStart(4, "0");
    return `${String(minutes).padStart(2, "0")}:${rest}`;
  }

  function seek(seconds) {
    audio.currentTime = Math.max(0, Math.min(duration, seconds));
    audio.play();
  }

  function swapAudioSource(source, time, wasPlaying) {
    audio.src = source;
    audio.load();
    audio.addEventListener("loadedmetadata", () => {
      audio.currentTime = Math.max(0, Math.min(duration, time));
      if (wasPlaying) audio.play();
    }, { once: true });
  }

  function updateSoloOptions() {
    const solo = meeting.solo || [];
    soloSelect.innerHTML = solo
      .map((speaker) => `<option value="${speaker.id}">${speaker.id} Â· ${speaker.raw} Â· ${speaker.speech.toFixed(1)}s</option>`)
      .join("");
    soloSelect.disabled = solo.length === 0;
    soloToggle.disabled = solo.length === 0;
    soloOn = false;
    soloToggle.textContent = "Play solo";
    soloToggle.classList.remove("active");
    if (solo.length) {
      soloSpeaker = solo[0];
      soloSelect.value = solo[0].id;
      const verification = meeting.soloVerification;
      const snr = verification && Number.isFinite(verification.snr_db)
        ? ` Â· reconstruction ${verification.snr_db.toFixed(0)} dB SNR`
        : "";
      soloStatus.textContent = `Exact source stems: clean target voice, all other speakers removed${snr}.`;
    } else {
      soloSpeaker = null;
      soloStatus.textContent = "No exact source stem is available for this meeting.";
    }
  }

  function setSolo(on) {
    if (!soloSpeaker) return;
    const time = audio.currentTime;
    const wasPlaying = !audio.paused;
    soloOn = on;
    swapAudioSource(on ? soloSpeaker.audio : meeting.audio, time, wasPlaying);
    soloToggle.textContent = on ? "Full mix" : "Play solo";
    soloToggle.classList.toggle("active", on);
    soloStatus.textContent = on
      ? `Solo: ${soloSpeaker.id} (${soloSpeaker.raw}) - only this speaker.`
      : `Full mixture. Exact solo stems available for ${meeting.solo.length} reference speakers.`;
  }

  function setPlayhead(seconds) {
    const fraction = Math.max(0, Math.min(1, seconds / duration));
    document.getElementById("wavePlayhead").style.left = `${fraction * 100}%`;
    const track = lanes.querySelector(".lane-track");
    if (track) {
      const trackRect = track.getBoundingClientRect();
      const lanesRect = lanes.getBoundingClientRect();
      playhead.style.left = `${trackRect.left - lanesRect.left + fraction * trackRect.width}px`;
    }
    document.getElementById("clock").textContent = `${fmt(seconds)} / ${fmt(duration)}`;
    document.querySelectorAll(".segment").forEach((element) => {
      const start = Number(element.dataset.start);
      const end = Number(element.dataset.end);
      element.classList.toggle("active", seconds >= start && seconds < end);
    });
  }

  function renderRuler() {
    const ruler = document.getElementById("ruler");
    const step = duration > 150 ? 30 : duration > 80 ? 20 : 10;
    let markup = "";
    for (let time = 0; time <= duration + 0.01; time += step) {
      markup += `<span class="tick" style="left:${(time / duration) * 100}%">${fmt(time)}</span>`;
    }
    if (duration % step > 0.5) {
      markup += `<span class="tick" style="left:100%">${fmt(duration)}</span>`;
    }
    ruler.innerHTML = markup;
  }

  function renderSystem(systemKey) {
    current = systemKey;
    const system = meeting.systems[systemKey];
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.system === systemKey);
    });

    const legend = document.getElementById("legend");
    legend.innerHTML = system.speakers
      .map((speaker) => `<span class="chip"><i style="background:${speaker.color}"></i>${speaker.id}<span class="mono muted">${speaker.raw}</span></span>`)
      .join("");

    lanes.innerHTML = "";
    for (const speaker of system.speakers) {
      const lane = document.createElement("div");
      lane.className = "lane";
      const label = document.createElement("div");
      label.className = "lane-label";
      label.innerHTML = `<i style="background:${speaker.color}"></i>${speaker.id}`;
      label.title = speaker.raw;
      const track = document.createElement("div");
      track.className = "lane-track";
      track.addEventListener("click", (event) => {
        const rect = track.getBoundingClientRect();
        seek(((event.clientX - rect.left) / rect.width) * duration);
      });
      for (const segment of system.segments.filter((item) => item.speaker === speaker.id)) {
        const element = document.createElement("div");
        element.className = "segment";
        element.style.left = `${(segment.start / duration) * 100}%`;
        element.style.width = `${Math.max(0.35, ((segment.end - segment.start) / duration) * 100)}%`;
        element.style.background = speaker.color;
        element.dataset.start = segment.start;
        element.dataset.end = segment.end;
        element.title = `${speaker.id} ${fmt(segment.start)} - ${fmt(segment.end)}`;
        element.addEventListener("click", (event) => {
          event.stopPropagation();
          seek(segment.start);
        });
        track.appendChild(element);
      }
      lane.append(label, track);
      lanes.appendChild(lane);
    }
    lanes.appendChild(playhead);

    const stats = system.stats;
    document.getElementById("systemMeta").innerHTML = `
      <div class="system-title">${system.label}</div>
      <div class="system-desc">${system.description}</div>
      <div class="system-score">${system.score ? `<strong>DER ${system.score.der.toFixed(2)}%</strong> &middot; JER ${system.score.jer.toFixed(2)}% &middot; B3-F1 ${system.score.b3_f1.toFixed(3)}` : "Reference RTTM; no score"}</div>
    `;
    document.getElementById("systemStats").innerHTML = `
      <div class="stat"><span>Speakers</span><strong>${stats.speakers}</strong></div>
      <div class="stat"><span>Turns</span><strong>${stats.turns}</strong></div>
      <div class="stat"><span>Speech</span><strong>${stats.speech.toFixed(1)}s</strong></div>
      <div class="stat"><span>Overlap</span><strong>${stats.overlap.toFixed(1)}s</strong></div>
    `;
    document.getElementById("systemLinks").innerHTML = `
      <a href="${system.rttm}">Open RTTM</a>
      ${system.score ? `<a href="score_${systemKey}/score.txt">Open score.txt</a>` : ""}
    `;

    const body = document.getElementById("segments");
    body.innerHTML = system.segments
      .map((segment, index) => `
        <tr data-start="${segment.start}">
          <td>${index + 1}</td>
          <td class="mono">${fmt(segment.start)}</td>
          <td class="mono">${fmt(segment.end)}</td>
          <td class="mono">${(segment.end - segment.start).toFixed(2)}s</td>
          <td><span class="chip"><i style="background:${segment.color}"></i>${segment.speaker}</span></td>
          <td class="mono muted">${segment.raw}</td>
        </tr>
      `)
      .join("");
    body.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", () => seek(Number(row.dataset.start)));
    });
    setPlayhead(audio.currentTime || 0);
  }

  function updateCards() {
    const ft = meeting.systems.finetuned.score;
    const base = meeting.systems.baseline.score;
    const ref = meeting.systems.reference.stats;
    const delta = base.der - ft.der;
    const deltaClass = delta >= 0 ? "good-text" : "bad-text";

    document.getElementById("cardFineDer").textContent = `${ft.der.toFixed(2)}%`;
    document.getElementById("cardFineNote").textContent = `JER ${ft.jer.toFixed(2)}% Â· B3-F1 ${ft.b3_f1.toFixed(3)}`;
    document.getElementById("cardBaseDer").textContent = `${base.der.toFixed(2)}%`;
    document.getElementById("cardBaseNote").textContent = `JER ${base.jer.toFixed(2)}% Â· B3-F1 ${base.b3_f1.toFixed(3)}`;
    const deltaNode = document.getElementById("cardDelta");
    deltaNode.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pt`;
    deltaNode.className = `card-value ${deltaClass}`;
    document.getElementById("cardSpeakers").textContent = `${ref.speakers}`;
    document.getElementById("cardTurns").textContent = `${ref.turns} turns`;
    document.getElementById("cardSpeech").textContent = `${ref.speech.toFixed(1)}s`;
    document.getElementById("cardOverlap2").textContent = `${ref.overlap.toFixed(1)}s`;
    document.getElementById("cardOverlapPct").textContent = `${ref.overlap_pct.toFixed(1)}% of file`;
    document.getElementById("fileGlanceNote").textContent = `Per-file DER on ${meeting.id}`;
  }

  function updateHeader() {
    const ft = meeting.systems.finetuned.score;
    const base = meeting.systems.baseline.score;
    const ref = meeting.systems.reference.stats;
    document.getElementById("badgeFine").textContent = `Fine-tuned ${DEMO.overall.finetuned.der.toFixed(2)}% DER`;
    document.getElementById("badgeBase").textContent = `Baseline ${DEMO.overall.baseline.der.toFixed(2)}% DER`;
    document.getElementById("badgeSpeakers").textContent = `${ref.speakers} speakers selected`;
    document.getElementById("badgeFiles").textContent = `${MEETINGS.length} dev meetings`;
    document.getElementById("meetingLabel").textContent = `${meeting.id} Â· ${meeting.durationFmt} Â· ${ref.speakers} speakers Â· ${ref.turns} turns`;
    document.getElementById("audioTitle").textContent = `Playable meeting audio Â· ${meeting.id}`;
  }

  function updateArtifacts() {
    const ft = meeting.systems.finetuned;
    const base = meeting.systems.baseline;
    const ref = meeting.systems.reference;
    document.getElementById("artifactList").innerHTML = `
      <div class="artifact"><code><a href="${meeting.audio}">${meeting.audio}</a></code><br><span>Playable 16 kHz mono WAV.</span></div>
      <div class="artifact"><code><a href="${ft.rttm}">fine-tuned RTTM</a></code><br><span>Predicted segments for this meeting.</span></div>
      <div class="artifact"><code><a href="${base.rttm}">baseline RTTM</a></code><br><span>Zero-shot prediction on the same audio.</span></div>
      <div class="artifact"><code><a href="${ref.rttm}">reference RTTM</a></code><br><span>Ground-truth segments from the dev split.</span></div>
      <div class="artifact"><code><a href="${meeting.ftScoreLink}">fine-tuned score.txt</a></code><br><span>10-file score table; per-file row included.</span></div>
      <div class="artifact"><code><a href="${meeting.baseScoreLink}">baseline score.txt</a></code><br><span>10-file score table; per-file row included.</span></div>
      ${meeting.solo && meeting.solo.length ? `<div class="artifact"><code>${meeting.solo.map((speaker) => `<a href="${speaker.audio}">${speaker.id} ${speaker.raw}</a>`).join(" Â· ")}</code><br><span>Exact target-only stems (PCM16 WAV).</span></div>` : ""}
    `;
  }

  function updateCallout() {
    const ft = meeting.systems.finetuned.score;
    const base = meeting.systems.baseline.score;
    const ref = meeting.systems.reference.stats;
    const delta = base.der - ft.der;
    const direction = delta >= 0 ? "improvement" : "regression";
    document.getElementById("callout").innerHTML = `
      <strong>Fine-tuned ${ft.der.toFixed(2)}% vs baseline ${base.der.toFixed(2)}% DER</strong>
      on this dev file: ${direction} of ${Math.abs(delta).toFixed(2)} points.
      Reference has ${ref.speakers} speakers, ${ref.turns} turns, and ${ref.overlap.toFixed(1)}s of overlap.
    `;
  }

  function updateTableSelection() {
    document.querySelectorAll("#meetingTable tr").forEach((row) => {
      row.classList.toggle("selected", row.dataset.id === meeting.id);
    });
  }

  function renderMeeting(meetingId) {
    const next = BY_ID[meetingId];
    if (!next) return;
    meeting = next;
    duration = meeting.duration;
    audio.pause();
    updateSoloOptions();
    audio.src = meeting.audio;
    audio.load();
    document.getElementById("waveform").src = meeting.waveform;
    document.getElementById("meetingSelect").value = meeting.id;
    document.getElementById("clock").textContent = `${fmt(0)} / ${fmt(duration)}`;
    updateHeader();
    updateCards();
    updateArtifacts();
    updateCallout();
    updateTableSelection();
    renderRuler();
    renderSystem(current);
  }

  function buildMeetingTable() {
    const body = document.getElementById("meetingTable");
    body.innerHTML = MEETINGS.map((item) => {
      const ft = item.systems.finetuned.score.der;
      const base = item.systems.baseline.score.der;
      const delta = base - ft;
      const deltaClass = delta >= 0 ? "good-text" : "bad-text";
      const speakers = item.systems.reference.stats.speakers;
      return `
        <tr data-id="${item.id}">
          <td class="mono">${item.id}</td>
          <td class="mono">${fmt(item.duration)}</td>
          <td>${speakers}</td>
          <td class="mono">${ft.toFixed(2)}%</td>
          <td class="mono">${base.toFixed(2)}%</td>
          <td class="mono ${deltaClass}">${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</td>
        </tr>
      `;
    }).join("");
    body.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", () => renderMeeting(row.dataset.id));
    });
  }

  function buildMeetingSelect() {
    const select = document.getElementById("meetingSelect");
    select.innerHTML = MEETINGS.map((item) => {
      const ft = item.systems.finetuned.score.der;
      const base = item.systems.baseline.score.der;
      return `<option value="${item.id}">${item.id} Â· FT ${ft.toFixed(2)}% / base ${base.toFixed(2)}%</option>`;
    }).join("");
    select.addEventListener("change", () => renderMeeting(select.value));
  }

  function stepMeeting(direction) {
    const index = MEETINGS.findIndex((item) => item.id === meeting.id);
    const next = MEETINGS[(index + direction + MEETINGS.length) % MEETINGS.length];
    renderMeeting(next.id);
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => renderSystem(tab.dataset.system));
  });
  soloSelect.addEventListener("change", () => {
    soloSpeaker = (meeting.solo || []).find((speaker) => speaker.id === soloSelect.value) || null;
    if (!soloSpeaker) return;
    if (soloOn) {
      const time = audio.currentTime;
      const wasPlaying = !audio.paused;
      swapAudioSource(soloSpeaker.audio, time, wasPlaying);
    }
    soloStatus.textContent = soloOn
      ? `Solo: ${soloSpeaker.id} (${soloSpeaker.raw}) - only this speaker.`
      : `Selected ${soloSpeaker.id} (${soloSpeaker.raw}). Press Play solo to isolate.`;
  });
  soloToggle.addEventListener("click", () => setSolo(!soloOn));
  document.getElementById("prevMeeting").addEventListener("click", () => stepMeeting(-1));
  document.getElementById("nextMeeting").addEventListener("click", () => stepMeeting(1));
  audio.addEventListener("timeupdate", () => setPlayhead(audio.currentTime));
  audio.addEventListener("seeking", () => setPlayhead(audio.currentTime));
  window.addEventListener("resize", () => setPlayhead(audio.currentTime || 0));

  buildMeetingSelect();
  buildMeetingTable();
  renderMeeting(meeting.id);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

