from __future__ import annotations
import sys
import subprocess,time,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

PY=sys.executable; SCRIPT=Path(__file__).resolve().parent / 'elastostatic_forward.py'; ROOT=Path(__file__).resolve().parent.parent / 'rerun_results' / 'reviewer11_experiments' / 'results' / 'sensitivity'; LOG=Path(__file__).resolve().parent.parent / 'rerun_results' / 'logs' / 'sensitivity'
base={'lr':5e-3,'qubits':5,'width':5,'n':1}
variants=[('lr1e-4',{'lr':1e-4}),('lr5e-4',{'lr':5e-4}),('lr5e-3',{'lr':5e-3}),('lr1e-3',{'lr':1e-3}),('lr2e-3',{'lr':2e-3}),('lr1e-2',{'lr':1e-2}),('q2',{'qubits':2}),('q4',{'qubits':4}),('q6',{'qubits':6}),('q10',{'qubits':10}),('w10',{'width':10}),('w20',{'width':20}),('n2',{'n':2}),('n3',{'n':3})]

def run(task):
    label,cfg,seed=task; out=ROOT/f'{label}_seed{seed}.json'; log=LOG/f'{label}_seed{seed}.out.log'; err=LOG/f'{label}_seed{seed}.err.log'
    if out.exists(): return label,seed,'skip'
    p=base|cfg; cmd=[PY,str(SCRIPT),'--model','qpinn','--bc','hard','--sampling','uniform','--weighting','fixed','--seed',str(seed),'--steps','500','--lr',str(p['lr']),'--qubits',str(p['qubits']),'--width',str(p['width']),'--n',str(p['n']),'--out',str(out)]
    with log.open('w',encoding='utf-8') as fo,err.open('w',encoding='utf-8') as fe: r=subprocess.run(cmd,stdout=fo,stderr=fe)
    return label,seed,r.returncode
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True); tasks=[(l,c,s) for l,c in variants for s in [0,1]]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs=[ex.submit(run,t) for t in tasks]
        for f in as_completed(futs): print(f.result(),flush=True)
