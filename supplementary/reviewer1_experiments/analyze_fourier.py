from __future__ import annotations
import json, math, statistics
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent / 'results' / 'fourier'
TARGETS={
 'f135': lambda x: torch.sin(x)+torch.sin(3*x)+torch.sin(5*x),
 'f1715': lambda x: 0.8*torch.sin(x)+0.4*torch.sin(7*x)+0.2*torch.sin(15*x),
}
FREQS={'f135':[1,3,5],'f1715':[1,7,15]}

def amp(signal,k): return float(torch.abs(torch.fft.rfft(torch.as_tensor(signal).reshape(-1)))[k])

def main():
    x=torch.linspace(-math.pi,math.pi,201)
    target_amp={t:[amp(TARGETS[t](x),k) for k in FREQS[t]] for t in TARGETS}
    groups={}
    for p in ROOT.glob('*.json'):
        if p.name.startswith('smoke_'): continue
        r=json.loads(p.read_text(encoding='utf-8')); key=(r['target'],r['model']); groups.setdefault(key,[]).append(r)
    rows=[]
    for (target,model),rs in sorted(groups.items()):
        rel=[float(r['relative_l2']) for r in rs]
        high_ratio=[float(r['final_amplitudes'][-1])/target_amp[target][-1] for r in rs]
        thresholds=[]
        for r in rs:
            hit=None
            for h in r['history']:
                if float(h['amplitudes'][-1]) >= 0.5*target_amp[target][-1]: hit=h['step']; break
            thresholds.append(hit if hit is not None else r['steps'])
        rows.append({'target':target,'model':model,'n':len(rs),'rel_l2_mean':statistics.mean(rel),'rel_l2_std':statistics.pstdev(rel) if len(rel)>1 else 0,'high_amp_ratio_mean':statistics.mean(high_ratio),'high_amp_ratio_std':statistics.pstdev(high_ratio) if len(high_ratio)>1 else 0,'steps_to_50pct_high_mean':statistics.mean(thresholds),'steps_to_50pct_high_std':statistics.pstdev(thresholds) if len(thresholds)>1 else 0,'params':int(rs[0]['params']),'steps':int(rs[0]['steps'])})
    out = Path(__file__).resolve().parent / 'fourier_summary.csv'
    if rows:
        with out.open('w',encoding='utf-8') as f:
            f.write(','.join(rows[0])+'\n'); f.writelines(','.join(str(r[k]) for k in rows[0])+'\n' for r in rows)
    for r in rows: print(r)
if __name__=='__main__': main()
