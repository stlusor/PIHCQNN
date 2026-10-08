from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn as nn

from common import FourierFeatureMLP, derivative, mlp, siren_mlp, count_parameters


def make_model(kind, case, seed, device):
    torch.manual_seed(seed)
    if case == 'poisson': in_dim,width=2,5
    elif case == 'heat': in_dim,width=2,10
    elif case == 'inverse': in_dim,width=1,5
    else: raise ValueError(case)
    if kind=='tanh2': return mlp(in_dim,width,2,1,'tanh').to(device)
    if kind=='tanh3': return mlp(in_dim,width,3,1,'tanh').to(device)
    if kind=='tanh5': return mlp(in_dim,width,3,1,'tanh').to(device)
    if kind=='tanh6': return mlp(in_dim,width,4,1,'tanh').to(device)
    if kind=='fourier': return FourierFeatureMLP(in_dim,width,3,32,4.0 if case=='heat' else 2.0).to(device)
    if kind in ('fourier05','fourier10','fourier20','fourier40'):
        sigma={'fourier05':0.5,'fourier10':1.0,'fourier20':2.0,'fourier40':4.0}[kind]
        return FourierFeatureMLP(in_dim,width,3,32,sigma).to(device)
    if kind=='siren': return siren_mlp(in_dim,width,3,1,5.0).to(device)
    if kind in ('siren1','siren5','siren30'):
        w0={'siren1':1.0,'siren5':5.0,'siren30':30.0}[kind]
        return siren_mlp(in_dim,width,3,1,w0).to(device)
    if kind=='adaptive': return mlp(in_dim,width,3,1,'adaptive_sin',0.5).to(device)
    if kind in ('adaptive01','adaptive05','adaptive10'):
        init={'adaptive01':0.1,'adaptive05':0.5,'adaptive10':1.0}[kind]
        return mlp(in_dim,width,3,1,'adaptive_sin',init).to(device)
    if kind=='lossbalance': return mlp(in_dim,width,3,1,'tanh').to(device)
    raise ValueError(kind)


def heat_source(t,x):
    t_max=.5; sigma=.02; p=.25*torch.cos(2*torch.pi*t/t_max)+.5
    pt=-.5*torch.sin(2*torch.pi*t/t_max)*torch.pi/t_max
    u=torch.exp(-(x-p)**2/(2*sigma**2)); k=.01*u+7; c=.0005*u**2+500; fac=1/sigma**2
    return fac*k*u+u*(x-p)*fac*(c*pt-(x-p)*fac*(k+.01*u))

def heat_exact(t,x):
    p=.25*torch.cos(4*torch.pi*t)+.5
    return torch.exp(-(x-p)**2/(2*.02**2))

def heat_data(n,device):
    t=torch.linspace(0,.5,n,device=device); x=torch.linspace(0,1,n,device=device); T,X=torch.meshgrid(t,x,indexing='ij'); T=T.reshape(-1,1).requires_grad_(); X=X.reshape(-1,1).requires_grad_()
    tb=torch.linspace(0,.5,20,device=device).reshape(-1,1); x0=torch.zeros_like(tb); x1=torch.ones_like(tb); xi=torch.linspace(0,1,20,device=device).reshape(-1,1); ti=torch.zeros_like(xi)
    for q in (tb,x0,x1,xi,ti): q.requires_grad_()
    return {'T':T,'X':X,'tb':tb,'x0':x0,'x1':x1,'xi':xi,'ti':ti}

def heat_losses(model,d):
    T,X=d['T'],d['X']; u=model(torch.cat([T,X],1)); ut=derivative(u,T); ux=derivative(u,X); uxx=derivative(ux,X)
    k=.01*u+7; c=.0005*u**2+500; lf=torch.mean((c*ut-.01*ux**2-k*uxx-heat_source(T,X))**2)
    p0=model(torch.cat([d['tb'],d['x0']],1)); p1=model(torch.cat([d['tb'],d['x1']],1))
    lb=torch.mean(derivative(p0,d['x0'])**2)+torch.mean(derivative(p1,d['x1'])**2)
    li=torch.mean((model(torch.cat([d['ti'],d['xi']],1))-heat_exact(d['ti'],d['xi']))**2)
    return [lf,lb,li]

def eval_heat(model,device):
    t=torch.linspace(0,.5,101,device=device); x=torch.linspace(0,1,201,device=device); T,X=torch.meshgrid(t,x,indexing='ij'); T=T.reshape(-1,1); X=X.reshape(-1,1)
    with torch.no_grad(): p=model(torch.cat([T,X],1)); y=heat_exact(T,X)
    return float(torch.linalg.norm(p-y)/torch.linalg.norm(y))

def poisson_data(n,device):
    z=torch.linspace(-1,1,n,device=device); X,Y=torch.meshgrid(z,z,indexing='ij'); X=X.reshape(-1,1).requires_grad_(); Y=Y.reshape(-1,1).requires_grad_()
    b=z.reshape(-1,1); bm=torch.full_like(b,-1); bp=torch.full_like(b,1)
    for q in (b,bm,bp): q.requires_grad_()
    return {'X':X,'Y':Y,'b':b,'bm':bm,'bp':bp}

def poisson_losses(model,d):
    X,Y=d['X'],d['Y']; u=model(torch.cat([X,Y],1)); uxx=derivative(u,X,2); uyy=derivative(u,Y,2); lf=torch.mean((uxx+uyy-X**2-Y**2)**2)
    bc=torch.cat([torch.cat([d['b'],d['bm']],1),torch.cat([d['b'],d['bp']],1),torch.cat([d['bm'],d['b']],1),torch.cat([d['bp'],d['b']],1)],0)
    target=torch.cat([.5*d['b']**2,.5*d['b']**2,.5*d['b']**2,.5*d['b']**2],0); lb=torch.mean((model(bc)-target)**2)
    return [lf,lb]

def eval_poisson(model,device):
    z=torch.linspace(-1,1,201,device=device); X,Y=torch.meshgrid(z,z,indexing='ij'); X=X.reshape(-1,1); Y=Y.reshape(-1,1)
    with torch.no_grad(): p=model(torch.cat([X,Y],1)); y=.5*(X*Y)**2
    return float(torch.linalg.norm(p-y)/torch.linalg.norm(y))

class InverseNet(nn.Module):
    def __init__(self,net): super().__init__(); self.net=net; self.EA=nn.Parameter(torch.tensor(2.0))
    def forward(self,x): return self.net(x)

def inverse_data(n,device):
    x=torch.linspace(0,1,n,device=device).reshape(-1,1).requires_grad_(); y=torch.sin(2*torch.pi*x)
    return {'x':x,'y':y}

def inverse_losses(model,d):
    x=d['x']
    # Separate graph for the PDE residual and data loss avoids reusing the same
    # graph after the nested autograd call used to form u_xx.
    u_pde=model(x); uxx=derivative(u_pde,x,2); lf=torch.mean((model.EA*uxx+4*torch.pi**2*torch.sin(2*torch.pi*x))**2)
    u_data=model(x); ld=torch.mean((u_data-d['y'])**2)
    return [lf,ld]

def eval_inverse(model,device):
    x=torch.linspace(0,1,201,device=device).reshape(-1,1)
    with torch.no_grad(): u=model(x); y=torch.sin(2*torch.pi*x)
    ea_err=float(torch.abs(model.EA-1)/1); uerr=float(torch.linalg.norm(u-y)/torch.linalg.norm(y)); return ea_err,uerr


def train(case,kind,seed,steps,grid,lr,device,batch=None):
    if case in ('heat','poisson'): model=make_model(kind,case,seed,device); data=heat_data(grid,device) if case=='heat' else poisson_data(grid,device)
    else: model=InverseNet(make_model(kind,case,seed,device)); data=inverse_data(grid,device)
    weights=([3e-7,1,1] if case=='heat' else [1,1] if case=='poisson' else [1,10])
    params=list(model.parameters())
    opt=torch.optim.Adam(params,lr=lr); hist=[]; start=time.perf_counter()
    adaptive_weights=list(weights)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if case=='heat':
            if batch and batch<data['T'].shape[0]:
                idx=torch.randint(0,data['T'].shape[0],(batch,),device=device); dd={k:v for k,v in data.items()}
                dd['T']=data['T'][idx].detach().requires_grad_(); dd['X']=data['X'][idx].detach().requires_grad_()
            else: dd=data
            losses=heat_losses(model,dd)
        elif case=='poisson':
            if batch and batch<data['X'].shape[0]:
                idx=torch.randint(0,data['X'].shape[0],(batch,),device=device); dd={k:v for k,v in data.items()}
                dd['X']=data['X'][idx].detach().requires_grad_(); dd['Y']=data['Y'][idx].detach().requires_grad_()
            else: dd=data
            losses=poisson_losses(model,dd)
        else: losses=inverse_losses(model,data)
        if kind=='lossbalance' and step % 500 == 0:
            norms=[]
            for li in losses:
                gs=torch.autograd.grad(li,params,retain_graph=True,allow_unused=True)
                norms.append(torch.sqrt(sum((g.square().sum() for g in gs if g is not None),torch.tensor(0.0,device=device))).item())
            target=sum(norms)/max(len(norms),1)
            adaptive_weights=[min(1e3,max(1e-3,target/(n+1e-12))) for n in norms]
        loss=sum(adaptive_weights[i]*losses[i] for i in range(len(losses)))
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            pct = 100.0 * (step + 1) / steps
            filled = int(pct // 5)
            bar = '#' * filled + '-' * (20 - filled)
            print(f'PROGRESS {case}/{kind} seed={seed} [{bar}] {pct:5.1f}% step={step+1}/{steps}', flush=True)
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        for p,g in zip(params,grads): p.grad=g
        opt.step()
        if step % 250 == 0 or step==steps-1:
            metric=(eval_heat(model,device) if case=='heat' else eval_poisson(model,device) if case=='poisson' else eval_inverse(model,device)[0])
            hist.append({'step':step+1,'loss':float(loss.detach().cpu()),'metric':metric})
    if case=='heat': metric=eval_heat(model,device); extra={}
    elif case=='poisson': metric=eval_poisson(model,device); extra={}
    else:
        ea_err,u_err=eval_inverse(model,device); metric=ea_err; extra={'EA':float(model.EA.detach().cpu()),'u_rel_l2':u_err}
    return {'case':case,'model':kind,'seed':seed,'steps':steps,'grid':grid,'params':count_parameters(model),'metric':metric,'metric_name':'relative_l2' if case!='inverse' else 'EA_relative_error','elapsed_s':time.perf_counter()-start,'history':hist,'extra':extra}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',choices=['heat','poisson','inverse'],required=True); ap.add_argument('--model',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int); ap.add_argument('--grid',type=int); ap.add_argument('--lr',type=float); ap.add_argument('--batch',type=int); ap.add_argument('--out')
    a=ap.parse_args(); defaults={'heat':(20000,50,2e-3),'poisson':(5000,40,2e-3),'inverse':(20000,50,5e-3)}[a.case]; steps=a.steps or defaults[0]; grid=a.grid or defaults[1]; lr=a.lr or defaults[2]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); r=train(a.case,a.model,a.seed,steps,grid,lr,device,a.batch); print(json.dumps(r,indent=2))
    if a.out: Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf-8')

if __name__=='__main__': main()
