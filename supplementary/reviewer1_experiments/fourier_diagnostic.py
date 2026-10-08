from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn as nn

from common import FourierFeatureMLP, PIHConfig, PIHCQNN, mlp, siren_mlp, count_parameters

TARGETS={
 'f135': lambda x: torch.sin(x)+torch.sin(3*x)+torch.sin(5*x),
 'f259': lambda x: torch.sin(2*x)+torch.sin(5*x)+torch.sin(9*x),
 'f1715': lambda x: 0.8*torch.sin(x)+0.4*torch.sin(7*x)+0.2*torch.sin(15*x),
}
TARGET_FREQS={'f135':[1,3,5],'f259':[2,5,9],'f1715':[1,7,15]}


def make_model(kind,seed,device):
    torch.manual_seed(seed)
    if kind.startswith('pih'):
        n=int(kind[-1]); return PIHCQNN(PIHConfig(1,20,10,n,1,False)).to(device)
    if kind=='fcnn5': return mlp(1,20,3,1,'tanh').to(device)
    if kind=='fcnn7': return mlp(1,20,5,1,'tanh').to(device)
    if kind=='fcnn8': return mlp(1,20,6,1,'tanh').to(device)
    if kind=='fourier': return FourierFeatureMLP(1,20,3,32,4.0).to(device)
    if kind=='siren': return siren_mlp(1,20,3,1,5.0).to(device)
    if kind=='adaptive': return mlp(1,20,3,1,'adaptive_sin',0.5).to(device)
    raise ValueError(kind)


def amplitude(pred,k):
    # Real FFT; for x in [-pi,pi] with 201 points, index k is the k-th harmonic.
    return float(torch.abs(torch.fft.rfft(pred.reshape(-1)))[k])


def run(kind,target_name,seed,steps,lr,device):
    x=torch.linspace(-math.pi,math.pi,201,device=device).reshape(-1,1)
    y=TARGETS[target_name](x)
    model=make_model(kind,seed,device); opt=torch.optim.Adam(model.parameters(),lr=lr)
    hist=[]; start=time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred=model(x); loss=torch.mean((pred-y)**2); loss.backward(); opt.step()
        if step%100==0 or step==steps-1:
            with torch.no_grad(): p=model(x); rel=float(torch.linalg.norm(p-y)/torch.linalg.norm(y)); amps=[amplitude(p,k) for k in TARGET_FREQS[target_name]]
            hist.append({'step':step+1,'loss':float(loss.detach().cpu()),'relative_l2':rel,'amplitudes':amps})
    with torch.no_grad(): p=model(x); rel=float(torch.linalg.norm(p-y)/torch.linalg.norm(y)); amps=[amplitude(p,k) for k in TARGET_FREQS[target_name]]
    return {'target':target_name,'model':kind,'seed':seed,'steps':steps,'params':count_parameters(model),'relative_l2':rel,'target_freqs':TARGET_FREQS[target_name],'final_amplitudes':amps,'elapsed_s':time.perf_counter()-start,'history':hist}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--target',choices=list(TARGETS),required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--out',required=True)
    a=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); r=run(a.model,a.target,a.seed,a.steps,a.lr,device); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in r.items() if k!='history'},indent=2))
if __name__=='__main__': main()
