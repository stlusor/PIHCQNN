from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
root = Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--results',default=str(root/'results'/'elastostatic_core'))
ap.add_argument('--outdir',default=str(root.parent/'rerun_results'/'training_analysis'))
a=ap.parse_args()
SOURCE=Path(a.results);OUT=Path(a.outdir);OUT.mkdir(parents=True,exist_ok=True)
rows = []
for p in sorted(SOURCE.glob('*.json')):
    if p.name.startswith('smoke'):
        continue
    j = json.loads(p.read_text(encoding='utf-8'))
    rows.append(j)
# Target u=sin(2*pi*x) on the 201-point evaluation grid.
x = [i / 200 for i in range(201)]
target = [math.sin(2 * math.pi * z) for z in x]
target_std = statistics.stdev(target)
groups = {}
for r in rows:
    k = (r['model'], r['bc'], r['sampling'], r['weighting'])
    groups.setdefault(k, []).append(r)
out = []
for k, rs in sorted(groups.items()):
    pstd = [r['pred_std'] for r in rs]
    pnorm = [r['pred_norm'] for r in rs]
    rel = [r['rel_l2'] for r in rs]
    out.append({
        'model': k[0], 'bc': k[1], 'sampling': k[2], 'weighting': k[3], 'n_runs': len(rs),
        'target_std': target_std,
        'pred_std_mean': statistics.mean(pstd),
        'pred_std_min': min(pstd),
        'pred_norm_mean': statistics.mean(pnorm),
        'pred_norm_min': min(pnorm),
        'rel_l2_mean': statistics.mean(rel),
        'rel_l2_max': max(rel),
        'collapse_runs_std_lt_10pct_target': sum(v < 0.1 * target_std for v in pstd),
    })
(OUT / 'reviewer11_trivial_solution_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
for r in out:
    print(r)
