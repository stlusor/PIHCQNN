from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
root = Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--results',default=str(root/'results'/'ansatz'))
ap.add_argument('--outdir',default=str(root.parent/'rerun_results'/'training_analysis'))
a=ap.parse_args()
SOURCE=Path(a.results);OUT=Path(a.outdir);OUT.mkdir(parents=True,exist_ok=True)
groups = {}
for p in sorted(SOURCE.glob('*.json')):
    j = json.loads(p.read_text(encoding='utf-8'))
    groups.setdefault(j['ansatz'], []).append(j)
out = []
for ansatz, rs in sorted(groups.items()):
    vals = [r['rel_l2'] for r in rs]
    out.append({
        'ansatz': ansatz,
        'n_runs': len(rs),
        'parameters': rs[0]['params'],
        'rel_l2_mean': statistics.mean(vals),
        'rel_l2_sd': statistics.stdev(vals) if len(vals) > 1 else 0.0,
        'rel_l2_min': min(vals),
        'rel_l2_max': max(vals),
        'elapsed_s_mean': statistics.mean(r['elapsed_s'] for r in rs),
    })
(OUT / 'reviewer11_ansatz_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
for r in out:
    print(r)
