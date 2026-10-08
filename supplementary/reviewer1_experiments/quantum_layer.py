from __future__ import annotations
import math
import torch
import torch.nn as nn
import pennylane as qml


def apply_rot(state: torch.Tensor, params: torch.Tensor, wire: int, nq: int) -> torch.Tensor:
    """Apply a PennyLane Rot(phi,theta,omega) to one wire of a batched statevector.
    state: [B, 2**nq], wire 0 is the least-significant amplitude bit.
    params: [..., 3] with components (phi, theta, omega).
    """
    phi, theta, omega = params[..., 0], params[..., 1], params[..., 2]
    if phi.ndim == 0:
        phi, theta, omega = phi.reshape(1), theta.reshape(1), omega.reshape(1)
    c = torch.cos(theta / 2).to(state.dtype)
    s = torch.sin(theta / 2).to(state.dtype)
    e_pm = torch.exp(-0.5j * (phi + omega)).to(state.dtype)
    e_pp = torch.exp(0.5j * (phi - omega)).to(state.dtype)
    e_mp = torch.exp(-0.5j * (phi - omega)).to(state.dtype)
    e_mm = torch.exp(0.5j * (phi + omega)).to(state.dtype)
    m00 = c * e_pm
    m01 = -s * e_pp
    m10 = s * e_mp
    m11 = c * e_mm
    low = 1 << wire
    high = 1 << (nq - wire - 1)
    x = state.reshape(state.shape[0], high, 2, low)
    x0 = x[:, :, 0, :]
    x1 = x[:, :, 1, :]
    def bc(v):
        return v.reshape(v.shape + (1,) * (x0.ndim - 1))
    y0 = bc(m00) * x0 + bc(m01) * x1
    y1 = bc(m10) * x0 + bc(m11) * x1
    return torch.stack((y0, y1), dim=2).reshape(state.shape[0], -1)


def apply_cnot(state: torch.Tensor, control: int, target: int, nq: int) -> torch.Tensor:
    """Apply CNOT to a batched statevector using index permutation (differentiable)."""
    base = torch.arange(1 << nq, device=state.device)
    mask = (base & (1 << control) != 0) & (base & (1 << target) == 0)
    src = base[mask]
    dst = src ^ (1 << target)
    a = state.index_select(1, src)
    b = state.index_select(1, dst)
    return state.index_copy(1, src, b).index_copy(1, dst, a)



_ENTANGLE_CACHE: dict[int, torch.Tensor] = {}


def entangle_permutation(nq: int, device: torch.device) -> torch.Tensor:
    key = nq
    cached = _ENTANGLE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    base = torch.arange(1 << nq, device=device)
    perm = base.clone()
    for control in range(nq):
        target = (control + 1) % nq
        src = base[(base & (1 << control) != 0) & (base & (1 << target) == 0)]
        dst = src ^ (1 << target)
        a = perm[src].clone()
        perm[src] = perm[dst]
        perm[dst] = a
    _ENTANGLE_CACHE[key] = perm
    return perm


def apply_strong_layer_fast(state: torch.Tensor, weights: torch.Tensor, nq: int) -> torch.Tensor:
    w = weights.reshape(nq, 3)
    for wire in range(nq):
        state = apply_rot(state, w[wire], wire, nq)
    return state.index_select(1, entangle_permutation(nq, state.device))


def apply_strong_layer(state: torch.Tensor, weights: torch.Tensor, nq: int) -> torch.Tensor:
    """One PennyLane StronglyEntanglingLayers layer with default range=1."""
    w = weights.reshape(nq, 3)
    for wire in range(nq):
        state = apply_rot(state, w[wire], wire, nq)
    for wire in range(nq):
        state = apply_cnot(state, wire, (wire + 1) % nq, nq)
    return state


def apply_angle_y(state: torch.Tensor, inputs: torch.Tensor, nq: int) -> torch.Tensor:
    for wire in range(nq):
        x = inputs[:, wire]
        c = torch.cos(x / 2).to(state.dtype)
        s = torch.sin(x / 2).to(state.dtype)
        low = 1 << wire
        high = 1 << (nq - wire - 1)
        z = state.reshape(state.shape[0], high, 2, low)
        z0, z1 = z[:, :, 0, :], z[:, :, 1, :]
        cb = c.reshape(c.shape + (1,) * (z0.ndim - 1))
        sb = s.reshape(s.shape + (1,) * (z0.ndim - 1))
        y0 = cb * z0 - sb * z1
        y1 = sb * z0 + cb * z1
        state = torch.stack((y0, y1), dim=2).reshape(state.shape[0], -1)
    return state


class ManualQVC(nn.Module):
    """Equivalent torch statevector implementation of the PIHCQNN VQC."""
    def __init__(self, n_qubits: int, n_rep: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rep = n_rep
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(n_qubits, 3)) for _ in range(n_rep + 1)
        ])
        for p in self.weights:
            nn.init.uniform_(p, 0.0, 2.0 * math.pi)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.n_qubits:
            raise ValueError(f'inputs must have shape [B,{self.n_qubits}]')
        state = torch.zeros((inputs.shape[0], 1 << self.n_qubits), dtype=torch.complex64, device=inputs.device)
        state[:, 0] = 1.0 + 0.0j
        for k in range(self.n_rep + 1):
            state = apply_strong_layer_fast(state, self.weights[k], self.n_qubits)
            if k < self.n_rep:
                state = apply_angle_y(state, inputs, self.n_qubits)
        probs = state.real ** 2 + state.imag ** 2
        # Sum all other bits after reshaping.  Easier and robust for small n: direct bit masks.
        base = torch.arange(1 << self.n_qubits, device=inputs.device)
        out = []
        for wire in range(self.n_qubits):
            bit = ((base >> wire) & 1) == 0
            p0 = probs[:, bit].sum(dim=1)
            p1 = probs[:, ~bit].sum(dim=1)
            out.append(p0 - p1)
        return torch.stack(out, dim=1)


def reference_qnodes(nq: int, n_rep: int):
    dev = qml.device('default.qubit', wires=nq)
    if n_rep == 1:
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def q(inputs, w0, w1):
            qml.StronglyEntanglingLayers(w0.reshape(1, nq, 3), wires=range(nq))
            qml.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
            qml.StronglyEntanglingLayers(w1.reshape(1, nq, 3), wires=range(nq))
            return [qml.expval(qml.PauliZ(i)) for i in range(nq)]
        return q, 2
    if n_rep == 2:
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def q(inputs, w0, w1, w2):
            qml.StronglyEntanglingLayers(w0.reshape(1, nq, 3), wires=range(nq))
            qml.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
            qml.StronglyEntanglingLayers(w1.reshape(1, nq, 3), wires=range(nq))
            qml.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
            qml.StronglyEntanglingLayers(w2.reshape(1, nq, 3), wires=range(nq))
            return [qml.expval(qml.PauliZ(i)) for i in range(nq)]
        return q, 3
    raise ValueError(n_rep)


def test():
    torch.manual_seed(7)
    for nq, nrep in [(2, 1), (3, 2), (5, 2)]:
        manual = ManualQVC(nq, nrep)
        q, nw = reference_qnodes(nq, nrep)
        x = torch.randn(4, nq)
        ref = q(x, *[p.reshape(1, nq, 3) for p in manual.weights])
        ref = torch.stack(ref, dim=1)
        out = manual(x)
        err = (ref - out).abs().max().item()
        print('compare', nq, nrep, 'maxerr', err, 'ref0', ref[0, :3].tolist(), 'out0', out[0, :3].tolist())
        assert err < 1e-5


if __name__ == '__main__':
    test()

