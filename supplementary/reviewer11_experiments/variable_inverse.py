from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'reviewer1_experiments'))
from common import PIHConfig,PIHCQNN,derivative,mlp
from elastostatic_forward import HardBC

THETA=torch.tensor([0.2,-0.1,0.05],dtype=torch.float64)
def basis(x): return torch.stack([torch.sin(2*math.pi*x),torch.sin(4*math.pi*x),torch.sin(6*math.pi*x)],dim=-1)
def coeff(x,theta): return 1.0+(basis(x)*theta).sum(dim=-1,keepdim=True)
def coeff_np(x,theta): return 1.0+theta[0]*np.sin(2*np.pi*x)+theta[1]*np.sin(4*np.pi*x)+theta[2]*np.sin(6*np.pi*x)
def source_np(x):
    t=THETA.numpy(); a=coeff_np(x,t); ap=2*np.pi*t[0]*np.cos(2*np.pi*x)+4*np.pi*t[1]*np.cos(4*np.pi*x)+6*np.pi*t[2]*np.cos(6*np.pi*x); ux=np.pi*np.cos(np.pi*x); uxx=-np.pi**2*np.sin(np.pi*x); return -(ap*ux+a*uxx)
def exact_u(x): return torch.sin(math.pi*x)
def source(x):
    ux=math.pi*torch.cos(math.pi*x); uxx=-math.pi**2*torch.sin(math.pi*x)
    bxp=torch.stack([2*math.pi*torch.cos(2*math.pi*x),4*math.pi*torch.cos(4*math.pi*x),6*math.pi*torch.cos(6*math.pi*x)],dim=-1)[:,0,:]
    ax=(bxp*THETA.to(x.device)).sum(dim=-1,keepdim=True); a=coeff(x,THETA.to(x.device))
    return -(ax*ux+a*uxx)

class InverseNet(torch.nn.Module):
    def __init__(self,base): super().__init__(); self.base=HardBC(base); self.theta=torch.nn.Parameter(torch.zeros(3,dtype=torch.float32))
    def forward(self,x): return self.base(x),self.theta

def build(model,width,qubits,n,seed,device):
    torch.manual_seed(seed)
    if model=='pinn': base=mlp(1,width,3,1,'tanh')
    else: base=PIHCQNN(PIHConfig(1,width,qubits,n,1,False))
    return InverseNet(base).to(device)

def train(model,width,qubits,n,seed,steps,lr,out):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); net=build(model,width,qubits,n,seed,device); x=torch.linspace(0,1,101,device=device).reshape(-1,1).requires_grad_(); xd=torch.linspace(0,1,50,device=device).reshape(-1,1); ud=exact_u(xd); params=list(net.parameters()); opt=torch.optim.Adam(params,lr=lr); t0=time.perf_counter()
    for step in range(steps):
        u,theta=net(x); ux=derivative(u,x); uxx=derivative(ux,x); a=coeff(x,theta); bxp=torch.stack([2*math.pi*torch.cos(2*math.pi*x),4*math.pi*torch.cos(4*math.pi*x),6*math.pi*torch.cos(6*math.pi*x)],dim=-1)[:,0,:]; ax=(bxp*theta).sum(dim=-1,keepdim=True)
        res=a*uxx+ax*ux+source(x); lf=res.pow(2).mean(); udp,_=net(xd); ld=(udp-ud).pow(2).mean(); loss=lf+10*ld
        opt.zero_grad(set_to_none=True); gs=torch.autograd.grad(loss,params,allow_unused=True)
        for p,g in zip(params,gs): p.grad=g
        opt.step()
        if step%max(1,steps//100)==0 or step==steps-1:
            pct=100*(step+1)/steps; bar='#'*int(pct//5)+'-'*(20-int(pct//5)); print(f'PROGRESS variable_inverse/{model} seed={seed} [{bar}] {pct:5.1f}% step={step+1}/{steps}',flush=True)
    xx=torch.linspace(0,1,201,device=device).reshape(-1,1); u,_=net(xx); theta=net.theta.detach().cpu(); result={'model':model,'seed':seed,'steps':steps,'theta_true':THETA.tolist(),'theta_hat':theta.tolist(),'theta_rel_error':float(torch.linalg.norm(theta-THETA)/torch.linalg.norm(THETA)),'u_rel_l2':float(torch.linalg.norm(u-exact_u(xx).to(device))/torch.linalg.norm(exact_u(xx).to(device))),'elapsed_s':time.perf_counter()-t0}; Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)

def fem_forward(theta,n=100):
    x=np.linspace(0,1,n+1); h=1/n; K=np.zeros((n+1,n+1)); F=np.zeros(n+1)
    for e in range(n):
        ae=coeff_np(np.array([.5*(x[e]+x[e+1])]),theta)[0]; ke=ae/h*np.array([[1.,-1.],[-1.,1.]])
        fe=h/2*np.array([source_np(x[e]),source_np(x[e+1])])
        for i in range(2):
            for j in range(2): K[e+i,e+j]+=ke[i,j]
            F[e+i]+=fe[i]
    K[0,:]=0; K[0,0]=1; F[0]=0; K[n,:]=0; K[n,n]=1; F[n]=0; return x,np.linalg.solve(K,F)

def fem_baseline():
    xobs=np.linspace(0,1,50); uobs=np.sin(np.pi*xobs)
    def residual(t):
        x,u=fem_forward(t,100); return np.interp(xobs,x,u)-uobs
    r=least_squares(residual,np.zeros(3),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    return {'model':'FEM_two_level','theta_hat':r.x.tolist(),'theta_rel_error':float(np.linalg.norm(r.x-THETA.numpy())/np.linalg.norm(THETA.numpy())),'nfev':int(r.nfev),'cost':float(r.cost)}
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['pinn','qpinn','fem'],required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--lr',type=float,default=5e-3); ap.add_argument('--width',type=int,default=5); ap.add_argument('--qubits',type=int,default=5); ap.add_argument('--n',type=int,default=1); ap.add_argument('--out',required=True); a=ap.parse_args()
    if a.model=='fem': r=fem_baseline(); Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf-8'); print(r)
    else: train(a.model,a.width,a.qubits,a.n,a.seed,a.steps,a.lr,a.out)
