from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reviewer1_experiments'))
from quantum_layer import ManualQVC


def main(a):
    torch.manual_seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]; repeats=a.repeats; batch=a.batch
    for nrep in a.n_reps:
        for nq in a.qubits:
            logs=[]; variances=[]
            for r in range(repeats):
                q=ManualQVC(nq,nrep).to(device)
                for p in q.weights: torch.nn.init.uniform_(p,0.0,2*math.pi)
                x=torch.rand(batch,nq,device=device)*2*math.pi
                y=q(x); loss=y.pow(2).mean(); grads=torch.autograd.grad(loss,list(q.parameters()),allow_unused=True)
                gn=math.sqrt(sum(float(g.detach().pow(2).sum().cpu()) for g in grads if g is not None)); logs.append(math.log10(max(gn,1e-300))); variances.append(float(y.var().detach().cpu()))
            mean=sum(logs)/len(logs); sd=(sum((v-mean)**2 for v in logs)/(len(logs)-1))**0.5
            rows.append({'n_rep':nrep,'qubits':nq,'repeats':repeats,'log10_grad_mean':mean,'log10_grad_sd':sd,'grad_norm_median':10**sorted(logs)[len(logs)//2],'output_variance_mean':sum(variances)/len(variances)})
            print(rows[-1],flush=True)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--repeats',type=int,default=50)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--n-reps',type=int,nargs='+',default=[1,2,3])
    ap.add_argument('--qubits',type=int,nargs='+',default=[2,4,6,8,10])
    ap.add_argument('--out',default=str(Path(__file__).resolve().parent/'results'/'gradient_scaling.json'))
    main(ap.parse_args())
