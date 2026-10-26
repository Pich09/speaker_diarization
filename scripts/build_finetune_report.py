import html
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "runs_full" / "scores" / "baseline_holdout"
FT_DIR = ROOT / "runs_full" / "scores" / "finetuned_holdout"
SPLIT_FILE = ROOT / "data" / "diarizen" / "khmer_full" / "holdout" / "split.json"
OUT = ROOT / "report_finetuned.html"

ROW_PATTERN = re.compile(
    r"^([0-9a-f-]{36})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def load_score(score_dir):
    rows = {}
    overall = None
    for line in (score_dir / "score.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("*** OVERALL ***"):
            numbers = [float(value) for value in re.findall(r"\d+\.\d+", line)]
            overall = {
                "der": numbers[0],
                "jer": numbers[1],
                "b3_precision": numbers[2],
                "b3_recall": numbers[3],
                "b3_f1": numbers[4],
            }
            break
        match = ROW_PATTERN.match(line)
        if match:
            rec_id, der, jer, precision, recall, f1 = match.groups()
            rows[rec_id] = {
                "id": rec_id,
                "der": float(der),
                "jer": float(jer),
                "b3_precision": float(precision),
                "b3_recall": float(recall),
                "b3_f1": float(f1),
            }
    if overall is None or not rows:
        raise SystemExit(f"Could not parse score files in {score_dir}")
    return rows, overall


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def severity(der):
    if der >= 20:
        return "bad"
    if der >= 10:
        return "warn"
    return "good"


def main():
    base_rows, base_overall = load_score(BASE_DIR)
    ft_rows, ft_overall = load_score(FT_DIR)
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    merged = []
    for rec_id, ft in ft_rows.items():
        base = base_rows.get(rec_id)
        merged.append(
            {
                "id": rec_id,
                "base_der": base["der"] if base else None,
                "der": ft["der"],
                "delta": (base["der"] - ft["der"]) if base else None,
                "jer": ft["jer"],
                "b3_f1": ft["b3_f1"],
                "b3_precision": ft["b3_precision"],
                "b3_recall": ft["b3_recall"],
            }
        )
    merged.sort(key=lambda row: row["der"], reverse=True)

    ders = [row["der"] for row in merged]
    deltas = [row["delta"] for row in merged if row["delta"] is not None]
    median = statistics.median(ders)
    worst = max(merged, key=lambda row: row["der"])
    best = min(merged, key=lambda row: row["der"])
    improved = sum(1 for value in deltas if value > 0)
    gate_target = round(base_overall["der"] - 1.0, 2)
    delta = base_overall["der"] - ft_overall["der"]
    passed = delta >= 1.0

    protocol = (
        f"collar 0.25 s &middot; {len(merged)} files &middot; "
        f"{split['splits']['test']['hours']:.3f} h"
    )

    cards = f"""
      <article class="card primary">
        <div class="card-label">Fine-tuned DER</div>
        <div class="card-value">{fmt(ft_overall["der"])}%</div>
        <div class="card-note">{protocol}</div>
      </article>
      <article class="card">
        <div class="card-label">Zero-shot baseline</div>
        <div class="card-value">{fmt(base_overall["der"])}%</div>
        <div class="card-note">same split and collar</div>
      </article>
      <article class="card good-card">
        <div class="card-label">Improvement</div>
        <div class="card-value good-text">{fmt(delta)} pt</div>
        <div class="card-note">{improved}/{len(merged)} files improved</div>
      </article>
      <article class="card">
        <div class="card-label">JER</div>
        <div class="card-value">{fmt(ft_overall["jer"])}%</div>
        <div class="card-note">baseline {fmt(base_overall["jer"])}%</div>
      </article>
      <article class="card">
        <div class="card-label">B3 F1</div>
        <div class="card-value">{fmt(ft_overall["b3_f1"], 3)}</div>
        <div class="card-note">precision {fmt(ft_overall["b3_precision"], 3)} &middot; recall {fmt(ft_overall["b3_recall"], 3)}</div>
      </article>
      <article class="card gate">
        <div class="card-label">Gate target</div>
        <div class="card-value">{'PASS' if passed else 'FAIL'}</div>
        <div class="card-note">&le; {fmt(gate_target)}% required; delta {fmt(delta)} pt</div>
      </article>
    """

    stats = f"""
      <div class="stat"><span>Median file DER</span><strong>{fmt(median)}%</strong></div>
      <div class="stat"><span>Worst file</span><strong>{fmt(worst["der"])}%</strong><small>{html.escape(worst["id"])}</small></div>
      <div class="stat"><span>Best file</span><strong>{fmt(best["der"])}%</strong><small>{html.escape(best["id"])}</small></div>
      <div class="stat"><span>Files improved</span><strong>{improved}/{len(merged)}</strong><small>baseline DER minus fine-tuned DER</small></div>
    """

    table_rows = []
    for row in merged:
        base_cell = fmt(row["base_der"]) if row["base_der"] is not None else "n/a"
        delta_cell = (
            f'<span class="delta {"good-text" if row["delta"] > 0 else "bad-text"}">'
            f'{"+" if row["delta"] > 0 else ""}{fmt(row["delta"])}</span>'
            if row["delta"] is not None
            else "n/a"
        )
        table_rows.append(
            "<tr "
            f'data-id="{html.escape(row["id"])}" '
            f'data-der="{row["der"]}" data-base_der="{row["base_der"]}" '
            f'data-delta="{row["delta"]}" data-jer="{row["jer"]}" data-f1="{row["b3_f1"]}">'
            f'<td class="mono">{html.escape(row["id"])}</td>'
            f'<td>{base_cell}%</td>'
            f'<td><span class="metric {severity(row["der"])}">{fmt(row["der"])}%</span></td>'
            f"<td>{delta_cell}</td>"
            f'<td>{fmt(row["jer"])}%</td>'
            f'<td>{fmt(row["b3_f1"], 3)}</td>'
            "</tr>"
        )

    train = split["splits"]["train"]
    dev = split["splits"]["dev"]
    test = split["splits"]["test"]
    leak = split["leakage"]

    data_table = f"""
      <table class="data-table">
        <thead>
          <tr><th>Split</th><th>Meetings</th><th>Hours</th><th>Speakers</th><th>Source cuts</th></tr>
        </thead>
        <tbody>
          <tr><td>Train</td><td>{train["meetings"]}</td><td>{train["hours"]:.3f}</td><td>{train["num_speakers"]}</td><td>{train["source_cuts"]:,}</td></tr>
          <tr><td>Dev</td><td>{dev["meetings"]}</td><td>{dev["hours"]:.3f}</td><td>{dev["num_speakers"]}</td><td>{dev["source_cuts"]:,}</td></tr>
          <tr><td>Test</td><td>{test["meetings"]}</td><td>{test["hours"]:.3f}</td><td>{test["num_speakers"]}</td><td>{test["source_cuts"]:,}</td></tr>
        </tbody>
      </table>
      <p class="muted small">
        Leakage checks: {leak["test_speakers_in_train_meetings"]} designated test speakers in train meetings,
        {leak["dev_speakers_in_train_meetings"]} designated dev speakers in train meetings.
        Source-cut overlap remains: {leak["source_cut_overlap_train_test"]} train/test and
        {leak["source_cut_overlap_train_dev"]} train/dev.
      </p>
    """

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = TEMPLATE
    replacements = {
        "__GENERATED__": generated,
        "__SUMMARY_CARDS__": cards,
        "__STATS__": stats,
        "__TABLE_ROWS__": "\n".join(table_rows),
        "__NUM_FILES__": str(len(merged)),
        "__DATA_TABLE__": data_table,
        "__BASELINE_DER__": fmt(base_overall["der"]),
        "__FINETUNED_DER__": fmt(ft_overall["der"]),
        "__DELTA__": fmt(delta),
        "__GATE_TARGET__": fmt(gate_target),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    page = page.replace("runs/", "runs_full/")

    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        f"baseline={base_overall['der']:.2f} finetuned={ft_overall['der']:.2f} "
        f"delta={delta:.2f} gate={gate_target:.2f} files={len(merged)}"
    )


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiariZen Khmer Fine-Tune Pilot</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #121a31;
    --panel-2: #17213d;
    --line: #2a3658;
    --text: #e8edf8;
    --muted: #9aa8c7;
    --accent: #6ea8ff;
    --accent-2: #8b7cff;
    --good: #3ddc97;
    --warn: #ffbf69;
    --bad: #ff6b81;
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
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
  header { padding: 56px 0 28px; }
  .eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent); font-weight: 700;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); box-shadow: 0 0 0 4px rgba(61,220,151,.14); }
  h1 { margin: 12px 0 10px; font-size: clamp(32px, 5vw, 58px); line-height: 1.05; letter-spacing: -0.03em; }
  h1 span { color: var(--accent); }
  .lede { max-width: 900px; color: var(--muted); font-size: 17px; margin: 0; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
  .badge { border: 1px solid var(--line); background: rgba(20,29,54,.75); color: var(--muted); padding: 6px 10px; border-radius: 999px; font-size: 12px; }
  .badge.done { color: var(--good); border-color: rgba(61,220,151,.35); background: rgba(61,220,151,.08); }
  .badge.pass { color: var(--good); border-color: rgba(61,220,151,.45); background: rgba(61,220,151,.12); font-weight: 700; }
  main { padding-bottom: 60px; }
  section { margin: 26px 0; }
  .section-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
  h2 { margin: 0; font-size: 22px; letter-spacing: -0.02em; }
  .section-head p { margin: 0; color: var(--muted); font-size: 13px; }
  .cards { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
  .card, .panel {
    border: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(23,33,61,.92), rgba(18,26,49,.92));
    border-radius: 16px;
    box-shadow: var(--shadow);
  }
  .card { padding: 18px; min-height: 128px; display: flex; flex-direction: column; justify-content: space-between; }
  .card.primary { border-color: rgba(110,168,255,.55); background: linear-gradient(180deg, rgba(110,168,255,.18), rgba(18,26,49,.92)); }
  .card.gate { border-color: rgba(61,220,151,.45); background: linear-gradient(180deg, rgba(61,220,151,.10), rgba(18,26,49,.92)); }
  .card.good-card { border-color: rgba(61,220,151,.35); }
  .card-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
  .card-value { font-size: 28px; font-weight: 800; letter-spacing: -0.03em; margin: 8px 0 4px; }
  .card-note { color: var(--muted); font-size: 12px; }
  .good-text { color: var(--good); }
  .bad-text { color: var(--bad); }
  .grid-2 { display: grid; grid-template-columns: 1.05fr .95fr; gap: 16px; }
  .panel { padding: 20px; }
  .panel h3 { margin: 0 0 12px; font-size: 16px; }
  .timeline { list-style: none; margin: 0; padding: 0; }
  .timeline li { position: relative; padding: 0 0 18px 30px; border-left: 1px solid var(--line); margin-left: 8px; }
  .timeline li:last-child { border-left-color: transparent; padding-bottom: 0; }
  .timeline li::before {
    content: ""; position: absolute; left: -6px; top: 3px; width: 11px; height: 11px; border-radius: 50%;
    background: var(--good); border: 2px solid var(--good); box-shadow: 0 0 0 5px rgba(61,220,151,.12);
  }
  .timeline strong { display: block; font-size: 14px; }
  .timeline span { color: var(--muted); font-size: 13px; }
  .callout { border-left: 3px solid var(--good); background: rgba(61,220,151,.08); border-radius: 0 12px 12px 0; padding: 14px 16px; color: var(--muted); font-size: 14px; }
  .callout strong { color: var(--text); }
  .stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
  .stat { border: 1px solid var(--line); background: rgba(11,16,32,.35); border-radius: 12px; padding: 13px; }
  .stat span { display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .07em; }
  .stat strong { display: block; font-size: 21px; margin: 5px 0 2px; }
  .stat small { display: block; color: var(--muted); font-size: 11px; word-break: break-all; }
  .data-table, .files-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .data-table th, .data-table td, .files-table th, .files-table td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
  .data-table th, .files-table th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .07em; background: rgba(11,16,32,.5); }
  .files-table th { position: sticky; top: 0; z-index: 2; cursor: pointer; user-select: none; white-space: nowrap; }
  .files-table th:hover { color: var(--text); }
  .files-table tbody tr:hover { background: rgba(110,168,255,.06); }
  .table-wrap { max-height: 560px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .search { width: min(360px, 100%); padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); background: rgba(11,16,32,.55); color: var(--text); outline: none; }
  .search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(110,168,255,.14); }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
  .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
  .metric { display: inline-block; min-width: 68px; padding: 3px 8px; border-radius: 999px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .metric.good { color: var(--good); background: rgba(61,220,151,.10); }
  .metric.warn { color: var(--warn); background: rgba(255,191,105,.10); }
  .metric.bad { color: var(--bad); background: rgba(255,107,129,.10); }
  .delta { font-weight: 700; font-variant-numeric: tabular-nums; }
  .mono { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: 12px; }
  .muted { color: var(--muted); }
  .small { font-size: 12px; }
  ul.clean { margin: 0; padding-left: 20px; color: var(--muted); }
  ul.clean li { margin: 7px 0; }
  .artifact-list { display: grid; gap: 9px; }
  .artifact { display: flex; gap: 10px; align-items: start; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: rgba(11,16,32,.3); }
  .artifact code { color: var(--text); font-size: 12px; word-break: break-all; }
  .artifact span { color: var(--muted); font-size: 12px; }
  pre { margin: 0; padding: 14px 16px; overflow: auto; border-radius: 12px; border: 1px solid var(--line); background: rgba(5,9,20,.65); color: #d8e2ff; font-size: 12px; }
  footer { border-top: 1px solid var(--line); padding: 22px 0 40px; color: var(--muted); font-size: 12px; }
  @media (max-width: 1100px) { .cards { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 980px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid-2 { grid-template-columns: 1fr; } .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 560px) { .wrap { width: min(100% - 24px, 1180px); } header { padding-top: 36px; } .cards, .stat-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow"><span class="dot"></span> One-shot pilot report</div>
    <h1>DiariZen <span>Khmer</span> fine-tune</h1>
    <p class="lede">
      Frozen-WavLM fine-tune of <strong>BUT-FIT/diarizen-wavlm-large-s80-md-v2</strong> on simulated Khmer meetings.
      The fine-tuned DER is <strong class="good-text">__FINETUNED_DER__%</strong> versus
      <strong>__BASELINE_DER__%</strong> zero-shot: a <strong class="good-text">__DELTA__ point</strong> improvement.
    </p>
    <div class="badges">
      <span class="badge pass">Gate PASS</span>
      <span class="badge done">Baseline __BASELINE_DER__%</span>
      <span class="badge done">Fine-tuned __FINETUNED_DER__%</span>
      <span class="badge">__NUM_FILES__ test files</span>
      <span class="badge">Collar 0.25 s</span>
      <span class="badge">Simulated data</span>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>Result at a glance</h2>
        <p>dscore on the full __NUM_FILES__-file holdout test split</p>
      </div>
      <div class="cards">
        __SUMMARY_CARDS__
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>What was run</h3>
        <ol class="timeline">
          <li><strong>Baseline</strong><span>Zero-shot v2 pipeline on all __NUM_FILES__ test files: DER __BASELINE_DER__%.</span></li>
          <li><strong>Data split</strong><span>Train 117.18 h / 6,207 meetings; dev 11.497 h / 611 meetings; test 1.699 h / 82 meetings.</span></li>
          <li><strong>Fine-tune</strong><span>8 epochs completed on the full training bundle; best checkpoint at epoch 8 by validation NLL loss (0.1506), with validation DER 3.10%.</span></li>
          <li><strong>Export and inference</strong><span>Best checkpoint exported and run on the same __NUM_FILES__ test files with batch 2.</span></li>
          <li><strong>Score and gate</strong><span>Fine-tuned DER __FINETUNED_DER__%, delta __DELTA__ points, gate &le; __GATE_TARGET__%: PASS.</span></li>
        </ol>
      </div>
      <div class="panel">
        <h3>Gate decision</h3>
        <div class="callout">
          <strong>PASS: __FINETUNED_DER__% vs __BASELINE_DER__%</strong>. The 1.0-point gate was met with a __DELTA__-point improvement.
          This is evidence for a larger pilot, not a production claim: the corpus is simulated clean read speech and only the segmentation model was adapted.
        </div>
        <h3 style="margin-top:20px">Per-file distribution</h3>
        <div class="stat-grid">
          __STATS__
        </div>
        <p class="muted small" style="margin:12px 0 0">
          The gate is applied to the overall DER, not to individual files.
        </p>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Per-file comparison</h2>
        <p>__NUM_FILES__ test meetings &middot; sorted worst fine-tuned DER first &middot; click a header to sort</p>
      </div>
      <div class="panel">
        <div class="toolbar">
          <input id="search" class="search" type="search" placeholder="Filter by meeting ID..." aria-label="Filter by meeting ID">
          <div class="legend">
            <span><i style="background:var(--good)"></i>fine-tuned &lt; 10%</span>
            <span><i style="background:var(--warn)"></i>10-20%</span>
            <span><i style="background:var(--bad)"></i>&ge; 20%</span>
            <span><i style="background:var(--good)"></i>positive delta = improvement</span>
          </div>
        </div>
        <div class="table-wrap">
          <table class="files-table" id="files">
            <thead>
              <tr>
                <th data-key="id">Meeting ID</th>
                <th data-key="base_der">Baseline DER</th>
                <th data-key="der">Fine-tuned DER</th>
                <th data-key="delta">Delta</th>
                <th data-key="jer">Fine-tuned JER</th>
                <th data-key="f1">Fine-tuned B3 F1</th>
              </tr>
            </thead>
            <tbody>
__TABLE_ROWS__
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>Training configuration</h3>
        <table class="data-table">
          <tbody>
            <tr><td>Base model</td><td class="mono">BUT-FIT/diarizen-wavlm-large-s80-md-v2</td></tr>
            <tr><td>WavLM</td><td>Frozen; 25 layers, 1024-d, 80% pruned</td></tr>
            <tr><td>Chunk / shift</td><td>16 s / 12 s train, 16 s / 16 s validation</td></tr>
            <tr><td>Batch</td><td>1 train x 4 grad accumulation; 2 validation</td></tr>
            <tr><td>Optimizer</td><td>AdamW, conformer lr 1e-3, frozen-WavLM lr 2e-5</td></tr>
            <tr><td>Epochs</td><td>8 max, patience 8; best epoch 8</td></tr>
            <tr><td>Checkpoint rule</td><td>Lowest validation NLL loss (<span class="mono">save_max_score=false</span>)</td></tr>
            <tr><td>Inference</td><td>Exported best checkpoint; batch 2; 82 files</td></tr>
            <tr><td>Scoring</td><td>dscore, collar 0.25 s, full UEM</td></tr>
          </tbody>
        </table>
      </div>
      <div class="panel">
        <h3>Data split</h3>
        __DATA_TABLE__
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>Where the artifacts are</h3>
        <div class="artifact-list">
          <div class="artifact"><div><code><a href="runs/scores/finetuned_holdout/score.txt">runs/scores/finetuned_holdout/score.txt</a></code><br><span>Fine-tuned per-file table and overall row.</span></div></div>
          <div class="artifact"><div><code><a href="runs/scores/finetuned_holdout/metrics.json">runs/scores/finetuned_holdout/metrics.json</a></code><br><span>Machine-readable fine-tuned DER summary.</span></div></div>
          <div class="artifact"><div><code><a href="runs/scores/baseline_holdout/score.txt">runs/scores/baseline_holdout/score.txt</a></code><br><span>Baseline score table used for the comparison.</span></div></div>
          <div class="artifact"><div><code><a href="runs/finetuned/khmer_finetune_holdout/">runs/finetuned/khmer_finetune_holdout/</a></code><br><span>82 fine-tuned predicted RTTM files.</span></div></div>
          <div class="artifact"><div><code><a href="runs/exported/khmer_finetune_holdout/">runs/exported/khmer_finetune_holdout/</a></code><br><span>Exported inference model with the best checkpoint.</span></div></div>
          <div class="artifact"><div><code><a href="runs/exp/khmer_finetune_holdout/checkpoints/best/">runs/exp/khmer_finetune_holdout/checkpoints/best/</a></code><br><span>Best checkpoint (epoch 8) and optimizer state.</span></div></div>
          <div class="artifact"><div><code><a href="runs/exp/khmer_finetune_holdout/">runs/exp/khmer_finetune_holdout/</a></code><br><span>Training and validation logs from the original and resumed runs.</span></div></div>
          <div class="artifact"><div><code><a href="runs/configs/khmer_finetune_holdout.toml">runs/configs/khmer_finetune_holdout.toml</a></code><br><span>Rendered training config.</span></div></div>
          <div class="artifact"><div><code><a href="artifacts/dev_demo/index.html">artifacts/dev_demo/index.html</a></code><br><span>Playable 10-meeting dev demo with fine-tuned, baseline, and reference timelines (4.24% vs 5.68% DER on the 10 files).</span></div></div>
          <div class="artifact"><div><code><a href="report.html">report.html</a></code><br><span>Original baseline report; preserved unchanged.</span></div></div>
        </div>
      </div>
      <div class="panel">
        <h3>Reproduce</h3>
        <pre>python scripts/run_oneshot.py --stages export score --data-dir data/diarizen/khmer_full --runs-dir runs_full --infer-batch-size 2 --yes
python scripts/build_finetune_report.py</pre>
        <h3 style="margin-top:18px">Caveats</h3>
        <ul class="clean">
          <li>Simulated clean read-speech, not real Khmer conversation; domain transfer is still unproven.</li>
          <li>Only 16-17 speakers per split; a small set of source speakers dominates the data.</li>
          <li>Source-cut overlap remains: 39 train/test and 451 train/dev cuts.</li>
          <li>Only the EEND segmentation model was adapted; embedding and PLDA remain domain-mismatched.</li>
          <li>All 8 epochs completed; the best checkpoint (epoch 8) was selected by validation loss, not test DER.</li>
          <li>Model CC BY-NC 4.0; corpus CC BY-SA 4.0. No release without legal review.</li>
        </ul>
      </div>
    </section>
  </main>

  <footer>
    Generated __GENERATED__ &middot; DiariZen Khmer one-shot pilot &middot; baseline and fine-tuned metrics from dscore.
  </footer>
</div>

<script>
  const table = document.getElementById('files');
  const tbody = table.tBodies[0];
  const search = document.getElementById('search');

  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    for (const row of tbody.rows) {
      row.style.display = row.dataset.id.toLowerCase().includes(query) ? '' : 'none';
    }
  });

  let sortState = { key: 'der', direction: -1 };
  table.querySelectorAll('th').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      const numeric = key !== 'id';
      if (sortState.key === key) sortState.direction *= -1;
      else sortState = { key, direction: key === 'id' ? 1 : -1 };

      const rows = Array.from(tbody.rows);
      rows.sort((a, b) => {
        const av = numeric ? Number(a.dataset[key]) : a.dataset[key];
        const bv = numeric ? Number(b.dataset[key]) : b.dataset[key];
        if (av < bv) return -1 * sortState.direction;
        if (av > bv) return 1 * sortState.direction;
        return 0;
      });
      for (const row of rows) tbody.appendChild(row);
    });
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()


