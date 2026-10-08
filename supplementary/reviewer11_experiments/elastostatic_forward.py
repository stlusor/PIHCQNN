from __future__ import annotations
import argparse, json, math, time, sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reviewer1_experiments"))
from common import PIHConfig, PIHCQNN, derivative, mlp


def source(x): return 4*math.pi**2*torch.sin(2*math.pi*x)
def exact(x): return torch.sin(2*math.pi*x)


class HardBC(nn.Module):
    def __init__(self, base): super().__init__(); self.base=base
    def forward(self,x): return x*(1-x)*self.base(x)


def build_model(model_kind,width,qubits,n,seed,device):
    torch.manual_seed(seed)
    if model_kind=='pinn': base=mlp(1,width,3,1,'tanh')
    elif model_kind=='qpinn': base=PIHCQNN(PIHConfig(1,width,qubits,n,1,False))
    else: raise ValueError(model_kind)
    return base.to(device)


def make_u(model,bc): return HardBC(model) if bc=='hard' else model
def residual(u,x): return derivative(u(x),x,2)+source(x)


def rel_l2(u,device,n=201):
    x=torch.linspace(0,1,n,device=device).reshape(-1,1)
    with torch.no_grad(): p=u(x); y=exact(x)
    return float(torch.linalg.norm(p-y)/torch.linalg.norm(y))


def sample_uniform(n,device): return torch.linspace(0,1,n+2,device=device)[1:-1].reshape(-1,1).requires_grad_()


def adaptive_points(u,device,n_candidates=1001,n_keep=100):
    x=torch.linspace(0,1,n_candidates,device=device)[1:-1].reshape(-1,1).requires_grad_()
    r=residual(u,x).detach().abs().reshape(-1)
    k=max(1,n_keep//2); top=torch.topk(r,k).indices
    rest=torch.ones_like(r,dtype=torch.bool); rest[top]=False
    pool=torch.nonzero(rest,as_tuple=False).reshape(-1)
    extra=pool[torch.randint(0,pool.numel(),(n_keep-k,),device=device)]
    idx=torch.cat([top,extra]); return x[idx].detach().reshape(-1,1).requires_grad_()


def group_norms(grads,params):
    q=0.0; c=0.0
    for g,(name,p) in zip(grads,params):
        if g is None: continue
        n=float(g.detach().pow(2).sum().sqrt().cpu())
        if 'q.weights' in name: q+=n*n
        else: c+=n*n
    return math.sqrt(q),math.sqrt(c)


def train(a):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); base=build_model(a.model,a.width,a.qubits,a.n,a.seed,device); u=make_u(base,a.bc).to(device)
    params=list(u.parameters()); opt=torch.optim.Adam(params,lr=a.lr)
    xf=sample_uniform(100,device) if a.sampling=='uniform' else adaptive_points(u,device,n_keep=100)
    wf=1.0; wb=1.0 if a.bc=='soft' else 0.0; hist=[]; t0=time.perf_counter(); named=list(u.named_parameters())
    for step in range(a.steps):
        if a.sampling=='adaptive' and step>0 and step%500==0: xf=adaptive_points(u,device,n_keep=100)
        lf=torch.mean(residual(u,xf)**2)
        if a.bc=='soft':
            x0=torch.zeros(1,1,device=device,requires_grad=True); x1=torch.ones(1,1,device=device,requires_grad=True); lb=u(x0).pow(2).mean()+u(x1).pow(2).mean()
        else: lb=torch.zeros((),device=device)
        opt.zero_grad(set_to_none=True)
        if a.weighting=='gradnorm' and step>0 and step%500==0 and a.bc=='soft':
            gl=torch.autograd.grad(lf,params,retain_graph=True,allow_unused=True)
            gb=torch.autograd.grad(lb,params,retain_graph=False,allow_unused=True)
            nl=math.sqrt(sum(float(g.detach().pow(2).sum().cpu()) for g in gl if g is not None)); nb=math.sqrt(sum(float(g.detach().pow(2).sum().cpu()) for g in gb if g is not None))
            target=(nl+nb)/2; wf=min(1e4,max(1e-4,target/(nl+1e-12))); wb=min(1e4,max(1e-4,target/(nb+1e-12)))
            grads=[(wf*(gl[i] if gl[i] is not None else torch.zeros_like(p))) + (wb*(gb[i] if gb[i] is not None else torch.zeros_like(p))) for i,p in enumerate(params)]
        else:
            loss=wf*lf+wb*lb
            grads=torch.autograd.grad(loss,params,allow_unused=True)
        for p,g in zip(params,grads): p.grad=g
        opt.step()
        if step%100==0 or step==a.steps-1:
            qn,cn=group_norms(grads,named); m=rel_l2(u,device,51)
            with torch.no_grad():
                xx=torch.linspace(0,1,201,device=device).reshape(-1,1); p=u(xx); pstd=float(p.std().cpu()); pnorm=float(p.norm().cpu()); pmean=float(p.mean().cpu())
            hist.append({'step':step+1,'loss':float(loss.detach().cpu()),'lf':float(lf.detach().cpu()),'lb':float(lb.detach().cpu()),'rel_l2_51':m,'grad_total':math.sqrt(qn*qn+cn*cn),'grad_q':qn,'grad_classical':cn,'pred_mean':pmean,'pred_std':pstd,'pred_norm':pnorm,'wf':wf,'wb':wb})
        if step%max(1,a.steps//100)==0 or step==a.steps-1:
            pct=100*(step+1)/a.steps; filled=int(pct//5); bar='#'*filled+'-'*(20-filled); print(f'PROGRESS {a.model}/{a.bc}/{a.sampling}/{a.weighting} seed={a.seed} [{bar}] {pct:5.1f}% step={step+1}/{a.steps}',flush=True)
    result={'case':'elastostatic_forward','model':a.model,'bc':a.bc,'sampling':a.sampling,'weighting':a.weighting,'seed':a.seed,'n':a.n,'qubits':a.qubits,'width':a.width,'lr':a.lr,'steps':a.steps,'params':sum(p.numel() for p in u.parameters()),'rel_l2':rel_l2(u,device,201),'elapsed_s':time.perf_counter()-t0,'history':hist}
    with torch.no_grad():
        x0=torch.tensor([[0.]],device=device); x1=torch.tensor([[1.]],device=device); result['boundary_l2']=(float(u(x0).pow(2).cpu())+float(u(x1).pow(2).cpu()))**0.5
        x=torch.linspace(0,1,201,device=device).reshape(-1,1); result['pred_mean']=float(u(x).mean().cpu()); result['pred_std']=float(u(x).std().cpu()); result['pred_norm']=float(u(x).norm().cpu())
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['pinn','qpinn'],required=True); ap.add_argument('--bc',choices=['soft','hard'],default='soft'); ap.add_argument('--sampling',choices=['uniform','adaptive'],default='uniform'); ap.add_argument('--weighting',choices=['fixed','gradnorm'],default='fixed'); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--lr',type=float,default=5e-3); ap.add_argument('--width',type=int,default=5); ap.add_argument('--qubits',type=int,default=5); ap.add_argument('--n',type=int,default=1); ap.add_argument('--out',required=True); train(ap.parse_args())
if __name__=='__main__': main()
