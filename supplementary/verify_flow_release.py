"""Verify sampling, archived predictions, and optionally the final model states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
FLOW = ROOT / 'flow_reproduction_20261007'
FINAL = {
    'pinn5': ROOT / 'reviewer_supplement_20261007/flow_controls/pinn5/physics_training',
    'pinn6': ROOT / 'reviewer_supplement_20261007/flow_controls/pinn6/physics_training',
    'pihcqnn': FLOW / 'convergence/pihcqnn_rms100_physics_graph',
}

def verify(checkpoints=False):
    protocol = json.loads((FLOW / 'protocol/protocol.json').read_text(encoding='utf-8'))
    points_path = FLOW / 'protocol/points.npz'
    assert hashlib.sha256(points_path.read_bytes()).hexdigest() == protocol['points_sha256']
    points = np.load(points_path)
    train, heldout = points['data_idx'], points['test_idx']
    expected = np.sort(np.random.RandomState(1234).choice(19340, 193, replace=False))
    np.testing.assert_array_equal(train, expected)
    np.testing.assert_array_equal(train, np.load(FLOW / 'protocol/data_indices.npy'))
    np.testing.assert_array_equal(heldout, np.load(FLOW / 'protocol/heldout_indices.npy'))
    assert len(heldout) == 19145
    _, group = np.unique(points['xy_ref'], axis=0, return_inverse=True)
    assert not np.intersect1d(group[train], group[heldout]).size
    records = []
    for name, folder in FINAL.items():
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        archive = np.load(folder / 'predictions.npz')
        np.testing.assert_array_equal(archive['heldout_idx'], heldout)
        np.testing.assert_array_equal(archive['data_idx'], train)
        np.testing.assert_array_equal(archive['xy'], points['xy_ref'])
        np.testing.assert_array_equal(archive['uvp_ref'], points['uvp_ref'])
        ref, pred = archive['uvp_ref'][heldout], archive['uvp_pred'][heldout]
        error = np.sqrt(((pred - ref)**2).sum(0) / (ref**2).sum(0))
        target = np.array([result['final_metrics']['heldout'][f'{s}_relative_l2'] for s in ['u', 'v', 'p']])
        np.testing.assert_allclose(error, target, rtol=1e-12, atol=1e-14)
        record = {'model': name, 'u_error_percent': float(100 * error[0]),
                  'v_error_percent': float(100 * error[1]),
                  'parameters': result['config']['parameters']}
        if checkpoints:
            import torch
            sys.path.insert(0, str(FLOW))
            from flow_core import make_model, evaluate, load_torch_checkpoint, set_runtime
            from convergence_train import NormalizedModel
            from convergence_quantum import replace_quantum
            set_runtime(1234)
            model = make_model(name)
            if name == 'pihcqnn':
                model = replace_quantum(model)
            model = NormalizedModel(model, True, True).to(dtype=torch.float64, device='cpu')
            state = load_torch_checkpoint(folder / 'checkpoint.pt')
            model.load_state_dict(state['model'])
            metrics, regenerated = evaluate(model, points['xy_ref'], points['uvp_ref'], device='cpu')
            np.testing.assert_allclose(regenerated, archive['uvp_pred'], rtol=2e-8, atol=1e-8)
            record['checkpoint_predictions_max_abs_difference'] = float(np.max(abs(regenerated - archive['uvp_pred'])))
            record['optimizer_outer_iterations'] = int(next(iter(state['optimizer']['state'].values()))['n_iter'])
        records.append(record)
    return {'sampling_verified': True, 'training_points': 193, 'heldout_points': 19145,
            'models': records, 'checkpoints_verified': checkpoints}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = verify(args.checkpoints)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
