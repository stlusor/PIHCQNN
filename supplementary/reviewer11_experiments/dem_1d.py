from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'reviewer1_experiments'))
from common import mlp
from elastostatic_forward import HardBC, exact, source

def train(seed,steps,lr,width,out):
    torch.manual_seed(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); base=mlp(1,width,3,1,'tanh').to(device); u=HardBC(base).to(device); x=torch.linspace(0,1,401,device=device).reshape(-1,1).requires_grad_(); params=list(u.parameters()); opt=torch.optim.Adam(params,lr=lr)
    for step in range(steps):
        ux=torch.autograd.grad(u(x),x,torch.ones_like(x),create_graph=True)[0]
        energy=torch.trapz(0.5*ux[:,0]**2-source(x)[:,0]*u(x)[:,0],x[:,0])
        opt.zero_grad(set_to_none=True); grads=torch.autograd.grad(energy,params,allow_unused=True)
        for p,g in zip(params,grads): p.grad=g
        opt.step()
    xx=torch.linspace(0,1,201,device=device).reshape(-1,1)
    with torch.no_grad(): p=u(xx); y=exact(xx)
    result={'model':'classical_DEM','seed':seed,'steps':steps,'width':width,'rel_l2':float(torch.linalg.norm(p-y)/torch.linalg.norm(y)),'params':sum(q.numel() for q in u.parameters()),'boundary_l2':0.0}
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--lr',type=float,default=5e-3); ap.add_argument('--width',type=int,default=5); ap.add_argument('--out',required=True); a=ap.parse_args(); train(a.seed,a.steps,a.lr,a.width,a.out)
