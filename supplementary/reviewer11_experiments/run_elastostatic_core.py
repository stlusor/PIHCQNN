from __future__ import annotations
import sys
import argparse, subprocess, itertools, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PY=sys.executable
SCRIPT=Path(__file__).resolve().parent / 'elastostatic_forward.py'
ROOT=Path(__file__).resolve().parent.parent / 'rerun_results' / 'reviewer11_experiments' / 'results' / 'elastostatic_core'
LOG=Path(__file__).resolve().parent.parent / 'rerun_results' / 'logs' / 'elastostatic_core'

CORE=[
 ('soft','uniform','fixed'),
 ('hard','uniform','fixed'),
 ('soft','uniform','gradnorm'),
 ('soft','adaptive','fixed'),
 ('hard','adaptive','fixed'),
]

def run(task):
    model,bc,sampling,weighting,seed=task
    name=f'{model}_{bc}_{sampling}_{weighting}_seed{seed}'
    out=ROOT/f'{name}.json'; log=LOG/f'{name}.out.log'; err=LOG/f'{name}.err.log'
    if out.exists(): return name,'skip'
    cmd=[PY,str(SCRIPT),'--model',model,'--bc',bc,'--sampling',sampling,'--weighting',weighting,'--seed',str(seed),'--steps','1000','--n','1','--qubits','5','--width','5','--lr','0.005','--out',str(out)]
    with log.open('w',encoding='utf-8') as fo, err.open('w',encoding='utf-8') as fe:
        r=subprocess.run(cmd,stdout=fo,stderr=fe)
    return name,r.returncode

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--seeds',type=int,default=3); a=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for model in ['pinn','qpinn']:
        for bc,sampling,weighting in CORE:
            for seed in range(a.seeds): tasks.append((model,bc,sampling,weighting,seed))
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs=[ex.submit(run,t) for t in tasks]
        for f in as_completed(futs): print(f.result(),flush=True)
if __name__=='__main__': main()
