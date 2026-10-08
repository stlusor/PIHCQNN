from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from variable_inverse import THETA, source_np


def assemble_fem(n: int = 100):
    x = np.linspace(0.0, 1.0, n + 1)
    h = 1.0 / n
    k0 = np.zeros((n + 1, n + 1))
    ks = [np.zeros((n + 1, n + 1)) for _ in range(3)]
    f = np.zeros(n + 1)
    elem = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for e in range(n):
        mid = 0.5 * (x[e] + x[e + 1])
        b = [np.sin(2 * np.pi * mid), np.sin(4 * np.pi * mid), np.sin(6 * np.pi * mid)]
        fe = h / 2.0 * np.array([source_np(x[e]), source_np(x[e + 1])])
        for i in range(2):
            for j in range(2):
                k0[e + i, e + j] += elem[i, j] / h
                for q in range(3):
                    ks[q][e + i, e + j] += b[q] * elem[i, j] / h
            f[e + i] += fe[i]
    for k in [k0, *ks]:
        k[0, :] = 0.0
        k[0, 0] = 1.0
        k[n, :] = 0.0
        k[n, n] = 1.0
    f[0] = 0.0
    f[n] = 0.0
    return x, k0, ks, f


def observation_operator(x: np.ndarray, n_obs: int = 50):
    xobs = np.linspace(0.0, 1.0, n_obs)
    idx = np.searchsorted(x, xobs) - 1
    idx = np.clip(idx, 0, len(x) - 2)
    w = (xobs - x[idx]) / (x[idx + 1] - x[idx])
    s = np.zeros((n_obs, len(x)))
    for i in range(n_obs):
        s[i, idx[i]] = 1.0 - w[i]
        s[i, idx[i] + 1] = w[i]
    return xobs, s


def run(a):
    x, k0, ks, f = assemble_fem(a.elements)
    xobs, s = observation_operator(x, a.observations)
    uobs = np.sin(np.pi * xobs)

    def objective_gradient(theta):
        k = k0.copy()
        for q in range(3):
            k += theta[q] * ks[q]
        u = np.linalg.solve(k, f)
        residual = s @ u - uobs
        objective = 0.5 * float(residual @ residual)
        adjoint = np.linalg.solve(k.T, s.T @ residual)
        gradient = np.array([-float(adjoint @ (ks[q] @ u)) for q in range(3)])
        return objective, gradient

    start = time.perf_counter()
    result = minimize(
        objective_gradient,
        np.zeros(3),
        jac=True,
        method='BFGS',
        options={'gtol': a.gtol, 'maxiter': a.maxiter},
    )
    elapsed = time.perf_counter() - start
    theta_true = THETA.numpy()
    out = {
        'model': 'FEM_two_level_adjoint',
        'theta_true': theta_true.tolist(),
        'theta_hat': result.x.tolist(),
        'theta_rel_error': float(np.linalg.norm(result.x - theta_true) / np.linalg.norm(theta_true)),
        'objective': float(result.fun),
        'success': bool(result.success),
        'message': result.message,
        'nfev': int(result.nfev),
        'elapsed_s': elapsed,
        'mesh_elements': a.elements,
        'observations': a.observations,
    }
    outpath = Path(a.out)
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--elements',type=int,default=100)
    ap.add_argument('--observations',type=int,default=50)
    ap.add_argument('--gtol',type=float,default=1e-12)
    ap.add_argument('--maxiter',type=int,default=1000)
    ap.add_argument('--out',default=str(Path(__file__).resolve().parent/'results'/'variable_inverse_fem_adjoint.json'))
    run(ap.parse_args())
