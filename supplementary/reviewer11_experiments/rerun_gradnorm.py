import sys
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
PY=sys.executable; SCRIPT=Path(__file__).resolve().parent / 'elastostatic_forward.py'; ROOT=Path(__file__).resolve().parent.parent / 'rerun_results' / 'reviewer11_experiments' / 'results' / 'elastostatic_core'; LOG=Path(__file__).resolve().parent.parent / 'rerun_results' / 'logs' / 'elastostatic_core'
def run(t):
 model,seed=t; out=ROOT/f'{model}_soft_uniform_gradnorm_seed{seed}.json'; log=LOG/f'{model}_soft_uniform_gradnorm_seed{seed}.out.log'; err=LOG/f'{model}_soft_uniform_gradnorm_seed{seed}.err.log'
 if out.exists(): return model,seed,'skip'
 cmd=[PY,str(SCRIPT),'--model',model,'--bc','soft','--sampling','uniform','--weighting','gradnorm','--seed',str(seed),'--steps','1000','--n','1','--qubits','5','--width','5','--lr','0.005','--out',str(out)]
 with log.open('w',encoding='utf-8') as fo,err.open('w',encoding='utf-8') as fe: r=subprocess.run(cmd,stdout=fo,stderr=fe)
 return model,seed,r.returncode
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
 tasks=[(m,s) for m in ['pinn','qpinn'] for s in [0,1,2]]
 with ThreadPoolExecutor(max_workers=3) as ex:
  for f in as_completed([ex.submit(run,t) for t in tasks]): print(f.result(),flush=True)
