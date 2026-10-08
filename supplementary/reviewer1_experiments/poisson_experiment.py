from __future__ import annotations
import math, time
from dataclasses import dataclass, asdict
from typing import Any
import torch
import torch.nn as nn

from common import ClassicalMLP, FourierFeatureMLP, PIHConfig, PIHCQNN, count_parameters, derivative, mlp, siren_mlp


def build_base_model(kind: str, case: str, seed: int, device: torch.device):
    torch.manual_seed(seed)
    if case == 'poisson':
        in_dim, width = 2, 5
    elif case == 'heat':
        in_dim, width = 2, 10
    elif case == 'inverse':
        in_dim, width = 1, 5
    elif case == 'fourier':
        in_dim = 1
    else:
        raise ValueError(case)

    if kind.startswith('pih'):
        nrep = int(kind[-1])
        cfg = PIHConfig(in_dim=in_dim, hidden_width=width, n_qubits=5 if case!='heat' else 10, n_rep=nrep)
        return PIHCQNN(cfg).to(device)
    if kind == 'tanh2':
        return mlp(in_dim,width,2,1,'tanh').to(device)
    if kind == 'tanh3':
        return mlp(in_dim,width,3,1,'tanh').to(device)
    if kind == 'tanh5':
        return mlp(in_dim,width,5,1,'tanh').to(device)
    if kind == 'fourier':
        return FourierFeatureMLP(in_dim,width,4,32,4.0 if case!='poisson' else 2.0).to(device)
    if kind == 'siren':
        return siren_mlp(in_dim,width,4,1,5.0).to(device)
    if kind == 'adaptive':
        return mlp(in_dim,width,4,1,'adaptive_sin',0.5).to(device)
    if kind == 'lossbalance':
        return mlp(in_dim,width,4,1,'tanh').to(device)
    raise ValueError(kind)


def build_case_model(kind: str, case: str, seed: int, device: torch.device):
    base = build_base_model(kind, case, seed, device)
    if case == 'inverse':
        return InverseModel(base).to(device)
    return base


class InverseModel(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__(); self.network=network; self.EA=nn.Parameter(torch.tensor(2.0))
    def forward(self,x): return self.network(x)


def poisson_data(grid: int, device):
    z=torch.linspace(-1,1,grid,device=device)
    X,Y=torch.meshgrid(z,z,indexing='ij'); X=X.reshape(-1,1); Y=Y.reshape(-1,1)
    X.requires_grad_(True); Y.requires_grad_(True)
    b=z.reshape(-1,1); bm=torch.full_like(b,-1); bp=torch.full_like(b,1)
    b.requires_grad_(True); bm.requires_grad_(True); bp.requires_grad_(True)
    return X,Y,b,bm,bp


def poisson_loss(model,data):
    X,Y,b,bm,bp=data
    u=model(torch.cat([X,Y],1)); uxx=derivative(u,X,2); uyy=derivative(u,Y,2)
    lf=torch.mean((uxx+uyy-X**2-Y**2)**2)
    bc=torch.cat([torch.cat([b,bm],1), torch.cat([b,bp],1), torch.cat([bm,b],1), torch.cat([bp,b],1)],0)
    target=torch.cat([0.5*b**2, 0.5*b**2, 0.5*b**2, 0.5*b**2],0)
    lb=torch.mean((model(bc)-target)**2)
    return lf,lb


def evaluate_poisson(model,device,n=201):
    z=torch.linspace(-1,1,n,device=device); X,Y=torch.meshgrid(z,z,indexing='ij'); X=X.reshape(-1,1); Y=Y.reshape(-1,1)
    with torch.no_grad(): pred=model(torch.cat([X,Y],1))
    true=0.5*(X*Y)**2
    return float(torch.linalg.norm(pred-true)/torch.linalg.norm(true))


def train_poisson(kind: str, seed: int, steps: int, grid: int, lr: float, device: torch.device, batch: int = 256):
    model=build_case_model(kind,'poisson',seed,device); data=poisson_data(grid,device)
    if kind=='lossbalance':
        base=model
        logs=None
    # Loss-balancing variant uses learnable global weights.
    logw=None
    if kind=='lossbalance':
        logw=nn.Parameter(torch.zeros(2,device=device))
        optim_params=list(model.parameters())+[logw]
    else:
        optim_params=list(model.parameters())
    opt=torch.optim.Adam(optim_params,lr=lr)
    hist=[]; start=time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        X,Y,b,bm,bp=data
        if batch and batch > 0:
            idx=torch.randint(0,X.shape[0],(batch,),device=device)
            Xb=X[idx].detach().requires_grad_(True); Yb=Y[idx].detach().requires_grad_(True)
            nbc=min(batch//4,b.shape[0]); ib=torch.randint(0,b.shape[0],(nbc,),device=device)
            bb=b[ib].detach().requires_grad_(True); bmb=bm[ib].detach().requires_grad_(True); bpb=bp[ib].detach().requires_grad_(True)
            lf,lb=poisson_loss(model,(Xb,Yb,bb,bmb,bpb))
        else:
            lf,lb=poisson_loss(model,data)
        if kind=='lossbalance':
            loss=torch.exp(logw[0])*lf+torch.exp(logw[1])*lb
        else:
            loss=lf+lb
        loss.backward(); opt.step()
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            pct = 100.0 * (step + 1) / steps
            filled = int(pct // 5)
            bar = '#' * filled + '-' * (20 - filled)
            print(f'PROGRESS poisson/{kind} seed={seed} [{bar}] {pct:5.1f}% step={step+1}/{steps}', flush=True)
        if step % 100 == 0 or step==steps-1:
            hist.append({'step':step+1,'loss':float(loss.detach().cpu()),'metric':evaluate_poisson(model,device,41)})
    metric=evaluate_poisson(model,device)
    params=count_parameters(model)+(0 if logw is None else logw.numel())
    return {'case':'poisson','model':kind,'seed':seed,'steps':steps,'grid':grid,'params':params,'metric':metric,'metric_name':'relative_l2','elapsed_s':time.perf_counter()-start,'history':hist}


if __name__=='__main__':
    import argparse, json
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--grid',type=int,default=40); ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--out')
    a=ap.parse_args(); r=train_poisson(a.model,a.seed,a.steps,a.grid,a.lr,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),a.batch); print(json.dumps(r,indent=2));
    if a.out: open(a.out,'w',encoding='utf-8').write(json.dumps(r,indent=2))
