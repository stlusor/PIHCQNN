from __future__ import annotations
import argparse, math, time, json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def rel_l2(pred, ref): return float(np.linalg.norm(pred-ref)/np.linalg.norm(ref))


def bar_fem(n):
    h=1.0/n; x=np.linspace(0,1,n+1); K=sparse.lil_matrix((n+1,n+1)); F=np.zeros(n+1)
    for e in range(n):
        Ke=(1/h)*np.array([[1.,-1.],[-1.,1.]])
        fe=h/2*np.array([4*math.pi**2*math.sin(2*math.pi*x[e]),4*math.pi**2*math.sin(2*math.pi*x[e+1])])
        for i in range(2):
            for j in range(2): K[e+i,e+j]+=Ke[i,j]
            F[e+i]+=fe[i]
    K=K.tocsr(); K[0,:]=0; K[0,0]=1; F[0]=0; K[n,:]=0; K[n,n]=1; F[n]=0
    return x,spsolve(K,F)


def poisson_fd(n):
    h=2.0/(n-1); x=np.linspace(-1,1,n); X,Y=np.meshgrid(x,x,indexing='ij'); idx=np.arange(n*n).reshape(n,n)
    A=sparse.lil_matrix((n*n,n*n)); b=np.zeros(n*n)
    for i in range(n):
        for j in range(n):
            k=idx[i,j]
            if i==0 or i==n-1 or j==0 or j==n-1:
                A[k,k]=1.0; b[k]=0.5*(X[i,j]*Y[i,j])**2
            else:
                A[k,k]=-4/h**2; A[k,k-1]=1/h**2; A[k,k+1]=1/h**2; A[k,k-n]=1/h**2; A[k,k+n]=1/h**2
                b[k]=X[i,j]**2+Y[i,j]**2
    U=spsolve(A.tocsr(),b).reshape(n,n); return X,Y,U


def heat_source_np(t,x):
    tmax=.5; sig=.02; p=.25*np.cos(2*np.pi*t/tmax)+.5; pt=-.5*np.sin(2*np.pi*t/tmax)*np.pi/tmax
    u=np.exp(-(x-p)**2/(2*sig**2)); k=.01*u+7; c=.0005*u**2+500; fac=1/sig**2
    return fac*k*u+u*(x-p)*fac*(c*pt-(x-p)*fac*(k+.01*u))

def heat_exact_np(t,x):
    p=.25*np.cos(4*np.pi*t)+.5; return np.exp(-(x-p)**2/(2*.02**2))

def heat_fd(nx=201,nt=10001):
    x=np.linspace(0,1,nx); dx=x[1]-x[0]; tmax=.5; dt=tmax/(nt-1); U=heat_exact_np(np.zeros_like(x),x)
    for it in range(1,nt):
        t=(it-1)*dt; u=U.copy(); uxx=np.zeros_like(u); uxx[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/dx**2; uxx[0]=uxx[1]; uxx[-1]=uxx[-2]
        k=.01*u+7; c=.0005*u**2+500
        ux=np.zeros_like(u); ux[1:-1]=(u[2:]-u[:-2])/(2*dx); ux[0]=0; ux[-1]=0
        rhs=.01*ux**2+k*uxx+heat_source_np(np.full_like(x,t),x)
        U=u+dt*rhs/c
        U[0]=U[1]; U[-1]=U[-2]
    return x,U,heat_exact_np(tmax*np.ones_like(x),x)


def main(a):
    out={}
    t=time.perf_counter(); x,u=bar_fem(a.bar_elements); out['bar_fem']={'n':a.bar_elements,'rel_l2':rel_l2(u,np.sin(2*np.pi*x)),'seconds':time.perf_counter()-t}
    t=time.perf_counter(); X,Y,U=poisson_fd(a.poisson_grid); out['poisson_fd']={'grid':a.poisson_grid,'rel_l2':rel_l2(U,.5*(X*Y)**2),'seconds':time.perf_counter()-t}
    t=time.perf_counter(); x,u,ref=heat_fd(a.heat_grid,a.heat_time_steps); out['heat_fd']={'grid':a.heat_grid,'time_steps':a.heat_time_steps,'rel_l2':rel_l2(u,ref),'seconds':time.perf_counter()-t}
    print(json.dumps(out,indent=2)); target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--bar-elements',type=int,default=200)
    ap.add_argument('--poisson-grid',type=int,default=201)
    ap.add_argument('--heat-grid',type=int,default=201)
    ap.add_argument('--heat-time-steps',type=int,default=10001)
    ap.add_argument('--out',default=str(Path(__file__).resolve().parent/'results'/'classical_baselines.json'))
    main(ap.parse_args())
