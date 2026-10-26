import html
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "runs" / "scores" / "baseline_holdout"
SPLIT_FILE = ROOT / "data" / "diarizen" / "khmer" / "holdout" / "split.json"
OUT = ROOT / "report.html"


def load_rows():
    rows = []
    overall = None
    pattern = re.compile(
        r"^([0-9a-f-]{36})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    for line in (SCORE_DIR / "score.txt").read_text(encoding="utf-8").splitlines():
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
        match = pattern.match(line)
        if match:
            rec_id, der, jer, precision, recall, f1 = match.groups()
            rows.append(
                {
                    "id": rec_id,
                    "der": float(der),
                    "jer": float(jer),
                    "b3_precision": float(precision),
                    "b3_recall": float(recall),
                    "b3_f1": float(f1),
                }
            )
    if overall is None:
        raise SystemExit("Could not parse *** OVERALL *** from score.txt")
    if not rows:
        raise SystemExit("Could not parse per-file rows from score.txt")
    return rows, overall


def severity(der):
    if der >= 20:
        return "bad"
    if der >= 10:
        return "warn"
    return "good"


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def main():
    rows, overall = load_rows()
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    ders = [row["der"] for row in rows]
    gate_target = round(overall["der"] - 1.0, 2)
    at_or_below_gate = sum(1 for value in ders if value <= gate_target)
    worst = max(rows, key=lambda row: row["der"])
    best = min(rows, key=lambda row: row["der"])
    median = statistics.median(ders)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    sorted_rows = sorted(rows, key=lambda row: row["der"], reverse=True)
    table_rows = []
    for row in sorted_rows:
        table_rows.append(
            "<tr "
            f'data-id="{html.escape(row["id"])}" '
            f'data-der="{row["der"]}" data-jer="{row["jer"]}" data-f1="{row["b3_f1"]}" '
            f'data-precision="{row["b3_precision"]}" data-recall="{row["b3_recall"]}">'
            f'<td class="mono">{html.escape(row["id"])}</td>'
            f'<td><span class="metric {severity(row["der"])}">{fmt(row["der"])}%</span></td>'
            f'<td>{fmt(row["jer"])}%</td>'
            f'<td>{fmt(row["b3_f1"], 3)}</td>'
            f'<td>{fmt(row["b3_precision"], 3)}</td>'
            f'<td>{fmt(row["b3_recall"], 3)}</td>'
            "</tr>"
        )

    cards = f"""
      <article class="card primary">
        <div class="card-label">Zero-shot baseline DER</div>
        <div class="card-value">{fmt(overall["der"])}%</div>
        <div class="card-note">collar 0.25 s Â· {len(rows)} files Â· {split["splits"]["test"]["hours"]:.3f} h</div>
      </article>
      <article class="card">
        <div class="card-label">JER</div>
        <div class="card-value">{fmt(overall["jer"])}%</div>
        <div class="card-note">Jaccard error rate</div>
      </article>
      <article class="card">
        <div class="card-label">B3 F1</div>
        <div class="card-value">{fmt(overall["b3_f1"], 3)}</div>
        <div class="card-note">precision {fmt(overall["b3_precision"], 3)} Â· recall {fmt(overall["b3_recall"], 3)}</div>
      </article>
      <article class="card gate">
        <div class="card-label">Gate target</div>
        <div class="card-value">&le; {fmt(gate_target)}%</div>
        <div class="card-note">fine-tuned DER must beat baseline by 1.0 point</div>
      </article>
      <article class="card">
        <div class="card-label">Fine-tuned result</div>
        <div class="card-value pending">Pending</div>
        <div class="card-note">no training run yet; `runs/exp` does not exist</div>
      </article>
    """

    stats = f"""
      <div class="stat"><span>Median file DER</span><strong>{fmt(median)}%</strong></div>
      <div class="stat"><span>Worst file</span><strong>{fmt(worst["der"])}%</strong><small>{html.escape(worst["id"])}</small></div>
      <div class="stat"><span>Best file</span><strong>{fmt(best["der"])}%</strong><small>{html.escape(best["id"])}</small></div>
      <div class="stat"><span>Files at/below gate</span><strong>{at_or_below_gate}/{len(rows)}</strong><small>not a substitute for overall DER</small></div>
    """

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
      <p class="muted">Leakage checks: {leak["test_speakers_in_train_meetings"]} designated test speakers in train meetings, {leak["dev_speakers_in_train_meetings"]} designated dev speakers in train meetings. Source-cut overlap remains: {leak["source_cut_overlap_train_test"]} train/test and {leak["source_cut_overlap_train_dev"]} train/dev.</p>
    """

    page = HTML_TEMPLATE
    replacements = {
        "__GENERATED__": generated,
        "__SUMMARY_CARDS__": cards,
        "__STATS__": stats,
        "__TABLE_ROWS__": "\n".join(table_rows),
        "__NUM_FILES__": str(len(rows)),
        "__DATA_TABLE__": data_table,
        "__OVERALL_DER__": fmt(overall["der"]),
        "__GATE_TARGET__": fmt(gate_target),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)

    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"DER={overall['der']:.2f} target={gate_target:.2f} files={len(rows)}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiariZen Khmer One-Shot Pilot</title>
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
  .lede { max-width: 850px; color: var(--muted); font-size: 17px; margin: 0; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
  .badge { border: 1px solid var(--line); background: rgba(20,29,54,.75); color: var(--muted); padding: 6px 10px; border-radius: 999px; font-size: 12px; }
  .badge.done { color: var(--good); border-color: rgba(61,220,151,.35); background: rgba(61,220,151,.08); }
  .badge.pending { color: var(--warn); border-color: rgba(255,191,105,.35); background: rgba(255,191,105,.08); }
  main { padding-bottom: 60px; }
  section { margin: 26px 0; }
  .section-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
  h2 { margin: 0; font-size: 22px; letter-spacing: -0.02em; }
  .section-head p { margin: 0; color: var(--muted); font-size: 13px; }
  .cards {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
  }
  .card, .panel {
    border: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(23,33,61,.92), rgba(18,26,49,.92));
    border-radius: 16px;
    box-shadow: var(--shadow);
  }
  .card { padding: 18px; min-height: 128px; display: flex; flex-direction: column; justify-content: space-between; }
  .card.primary { border-color: rgba(110,168,255,.55); background: linear-gradient(180deg, rgba(110,168,255,.18), rgba(18,26,49,.92)); }
  .card.gate { border-color: rgba(255,191,105,.45); }
  .card-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
  .card-value { font-size: 30px; font-weight: 800; letter-spacing: -0.03em; margin: 8px 0 4px; }
  .card-value.pending { color: var(--warn); font-size: 24px; }
  .card-note { color: var(--muted); font-size: 12px; }
  .grid-2 { display: grid; grid-template-columns: 1.05fr .95fr; gap: 16px; }
  .panel { padding: 20px; }
  .panel h3 { margin: 0 0 12px; font-size: 16px; }
  .timeline { list-style: none; margin: 0; padding: 0; }
  .timeline li { position: relative; padding: 0 0 18px 30px; border-left: 1px solid var(--line); margin-left: 8px; }
  .timeline li:last-child { border-left-color: transparent; padding-bottom: 0; }
  .timeline li::before {
    content: ""; position: absolute; left: -6px; top: 3px; width: 11px; height: 11px; border-radius: 50%;
    background: var(--panel-2); border: 2px solid var(--line);
  }
  .timeline li.done::before { background: var(--good); border-color: var(--good); box-shadow: 0 0 0 5px rgba(61,220,151,.12); }
  .timeline li.pending::before { background: var(--warn); border-color: var(--warn); box-shadow: 0 0 0 5px rgba(255,191,105,.10); }
  .timeline strong { display: block; font-size: 14px; }
  .timeline span { color: var(--muted); font-size: 13px; }
  .callout { border-left: 3px solid var(--accent); background: rgba(110,168,255,.08); border-radius: 0 12px 12px 0; padding: 14px 16px; color: var(--muted); font-size: 14px; }
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
  .mono { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: 12px; }
  .muted { color: var(--muted); }
  ul.clean { margin: 0; padding-left: 20px; color: var(--muted); }
  ul.clean li { margin: 7px 0; }
  .artifact-list { display: grid; gap: 9px; }
  .artifact { display: flex; gap: 10px; align-items: start; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: rgba(11,16,32,.3); }
  .artifact code { color: var(--text); font-size: 12px; word-break: break-all; }
  .artifact span { color: var(--muted); font-size: 12px; }
  pre { margin: 0; padding: 14px 16px; overflow: auto; border-radius: 12px; border: 1px solid var(--line); background: rgba(5,9,20,.65); color: #d8e2ff; font-size: 12px; }
  footer { border-top: 1px solid var(--line); padding: 22px 0 40px; color: var(--muted); font-size: 12px; }
  @media (max-width: 980px) {
    .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-2 { grid-template-columns: 1fr; }
    .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 560px) {
    .wrap { width: min(100% - 24px, 1180px); }
    header { padding-top: 36px; }
    .cards, .stat-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow"><span class="dot"></span> One-shot pilot report</div>
    <h1>DiariZen <span>Khmer</span> baseline</h1>
    <p class="lede">A gated pilot for fine-tuning <strong>BUT-FIT/diarizen-wavlm-large-s80-md-v2</strong> on simulated Khmer meetings. The zero-shot baseline is complete; training has not started.</p>
    <div class="badges">
      <span class="badge done">Baseline complete</span>
      <span class="badge pending">Fine-tune pending</span>
      <span class="badge">82 test files</span>
      <span class="badge">Collar 0.25 s</span>
      <span class="badge">Simulated data</span>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>Result at a glance</h2>
        <p>dscore on the full 82-file holdout test split</p>
      </div>
      <div class="cards">
        __SUMMARY_CARDS__
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>One-shot process</h3>
        <ol class="timeline">
          <li class="done"><strong>Data audit</strong><span>6,900 simulated meetings, 130.37 h, 20 speakers; clean read-speech, no noise or reverb.</span></li>
          <li class="done"><strong>Environment setup</strong><span>DiariZen + pyannote-audio, Windows audio fallback, PyTorch embedding loading, Perl for dscore.</span></li>
          <li class="done"><strong>Model and checkpoint</strong><span>v2 model plus WeSpeaker ResNet34 embedding; 599/599 checkpoint keys matched.</span></li>
          <li class="done"><strong>Speaker-disjoint holdout</strong><span>Train 1.027 h, dev 11.497 h, test 82 meetings / 1.699 h; designated test speakers excluded from train meetings.</span></li>
          <li class="done"><strong>Zero-shot baseline</strong><span>Diarized all 82 test files with the v2 pipeline and wrote per-file RTTMs.</span></li>
          <li class="done"><strong>Score with dscore</strong><span>Overall DER __OVERALL_DER__% at 0.25 s collar; JER and B3 metrics also recorded.</span></li>
          <li class="pending"><strong>Frozen-WavLM pilot</strong><span>Not started. Gate target is DER &le; __GATE_TARGET__%. A 4 GB GPU VRAM probe is the next step.</span></li>
        </ol>
      </div>
      <div class="panel">
        <h3>What â€œone-shotâ€ means</h3>
        <div class="callout">
          <strong>One gated pilot experiment</strong>, not one-shot speaker enrollment or a production model. Passing the gate only justifies building a real Khmer evaluation set and a larger pilot.
        </div>
        <h3 style="margin-top:20px">Distribution</h3>
        <div class="stat-grid">
          __STATS__
        </div>
        <p class="muted" style="font-size:12px;margin:12px 0 0">The gate is applied to the overall DER, not to individual files. â€œFiles at/below gateâ€ is descriptive only.</p>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Per-file DER</h2>
        <p>__NUM_FILES__ test meetings Â· sorted worst first Â· click a header to sort</p>
      </div>
      <div class="panel">
        <div class="toolbar">
          <input id="search" class="search" type="search" placeholder="Filter by meeting IDâ€¦" aria-label="Filter by meeting ID">
          <div class="legend">
            <span><i style="background:var(--good)"></i>&lt; 10%</span>
            <span><i style="background:var(--warn)"></i>10â€“20%</span>
            <span><i style="background:var(--bad)"></i>&ge; 20%</span>
          </div>
        </div>
        <div class="table-wrap">
          <table class="files-table" id="files">
            <thead>
              <tr>
                <th data-key="id">Meeting ID</th>
                <th data-key="der">DER</th>
                <th data-key="jer">JER</th>
                <th data-key="f1">B3 F1</th>
                <th data-key="precision">B3 precision</th>
                <th data-key="recall">B3 recall</th>
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
        <h3>Data split</h3>
        __DATA_TABLE__
      </div>
      <div class="panel">
        <h3>Model and evaluation</h3>
        <table class="data-table">
          <tbody>
            <tr><td>Model</td><td class="mono">BUT-FIT/diarizen-wavlm-large-s80-md-v2</td></tr>
            <tr><td>Backbone</td><td>WavLM-Large, 80% pruned, 63.3M params</td></tr>
            <tr><td>Segmentation</td><td>16 s chunks, 0.1 s step, max 4 speakers/frame</td></tr>
            <tr><td>Clustering</td><td>VBx + PLDA, 1â€“20 speakers</td></tr>
            <tr><td>Embedding</td><td>WeSpeaker VoxCeleb ResNet34</td></tr>
            <tr><td>Scoring</td><td>dscore, collar 0.25 s, full UEM</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h3>Where the artifacts are</h3>
        <div class="artifact-list">
          <div class="artifact"><div><code><a href="runs/scores/baseline_holdout/score.txt">runs/scores/baseline_holdout/score.txt</a></code><br><span>Human-readable per-file table and overall row.</span></div></div>
          <div class="artifact"><div><code><a href="runs/scores/baseline_holdout/metrics.json">runs/scores/baseline_holdout/metrics.json</a></code><br><span>Machine-readable DER summary.</span></div></div>
          <div class="artifact"><div><code><a href="runs/zero_shot/holdout/">runs/zero_shot/holdout/</a></code><br><span>82 predicted RTTM files.</span></div></div>
          <div class="artifact"><div><code><a href="runs/configs/khmer_finetune_holdout.toml">runs/configs/khmer_finetune_holdout.toml</a></code><br><span>Rendered frozen-WavLM training config.</span></div></div>
          <div class="artifact"><div><code><a href="data/diarizen/khmer/holdout/">data/diarizen/khmer/holdout/</a></code><br><span>Train/dev/test bundles, reference RTTMs, UEM, split report.</span></div></div>
        </div>
      </div>
      <div class="panel">
        <h3>Reproduce</h3>
        <pre>python scripts/run_oneshot.py --stages baseline --yes
python scripts/run_oneshot.py --stages score --yes

# view the overall row
Get-Content runs\scores\baseline_holdout\score.txt |
  Select-String "OVERALL"</pre>
        <h3 style="margin-top:18px">Caveats</h3>
        <ul class="clean">
          <li>Simulated clean read-speech, not real Khmer conversation; domain transfer is likely poor.</li>
          <li>Only 20 speakers; top-5 speakers dominate the training data.</li>
          <li>Source-cut overlap between train and test remains (3 train/test overlaps).</li>
          <li>Fine-tuning adapts segmentation only; embedding and PLDA stay domain-mismatched.</li>
          <li>Model CC BY-NC 4.0; corpus CC BY-SA 4.0. No release without legal review.</li>
        </ul>
      </div>
    </section>
  </main>

  <footer>
    Generated __GENERATED__ Â· DiariZen Khmer one-shot pilot Â· baseline metric from dscore.
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


