from __future__ import annotations
import argparse,json,statistics,csv
from pathlib import Path
SCRIPT_ROOT=Path(__file__).resolve().parent
ROOT=SCRIPT_ROOT/'results'
OUT=SCRIPT_ROOT.parent/'rerun_results'/'training_analysis'

def rows_from(pattern):
 out=[]
 for p in ROOT.glob(pattern):
  if p.name.startswith('smoke'): continue
  try: out.append(json.loads(p.read_text(encoding='utf-8')))
  except: pass
 return out

def summarize_core():
 rows=rows_from('elastostatic_core/*.json'); groups={}
 for r in rows:
  k=(r['model'],r['bc'],r['sampling'],r['weighting']); groups.setdefault(k,[]).append(r)
 out=[]
 for k,rs in sorted(groups.items()):
  vals=[r['rel_l2'] for r in rs]; out.append({'model':k[0],'bc':k[1],'sampling':k[2],'weighting':k[3],'n':len(rs),'rel_l2_mean':statistics.mean(vals),'rel_l2_sd':statistics.stdev(vals) if len(vals)>1 else 0,'boundary_mean':statistics.mean(r['boundary_l2'] for r in rs),'pred_std_mean':statistics.mean(r['pred_std'] for r in rs)})
 return out

def summarize_sensitivity():
 rows=rows_from('sensitivity/*.json'); groups={}
 for r in rows:
  label=None
  # derive from file later; use config tuple
  k=(r['lr'],r['qubits'],r['width'],r['n']); groups.setdefault(k,[]).append(r)
 out=[]
 for k,rs in sorted(groups.items()):
  vals=[r['rel_l2'] for r in rs]; out.append({'lr':k[0],'qubits':k[1],'width':k[2],'n':k[3],'n_runs':len(rs),'rel_l2_mean':statistics.mean(vals),'rel_l2_sd':statistics.stdev(vals) if len(vals)>1 else 0})
 return out

def summarize_variable():
 rows=rows_from('variable_inverse_*.json'); groups={}
 for r in rows: groups.setdefault(r['model'],[]).append(r)
 out=[]
 for model,rs in groups.items():
  out.append({'model':model,'n':len(rs),'theta_rel_mean':statistics.mean(r['theta_rel_error'] for r in rs),'theta_rel_sd':statistics.stdev(r['theta_rel_error'] for r in rs) if len(rs)>1 else 0,'u_rel_mean':statistics.mean(r['u_rel_l2'] for r in rs) if 'u_rel_l2' in rs[0] else None})
 return out

def main():
 core=summarize_core(); sens=summarize_sensitivity(); vinv=summarize_variable();
 for name,data in [('reviewer11_core_summary.json',core),('reviewer11_sensitivity_summary.json',sens),('reviewer11_variable_inverse_summary.json',vinv)]: (OUT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
 print('CORE'); [print(r) for r in core]; print('SENS'); [print(r) for r in sens]; print('VAR'); [print(r) for r in vinv]
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--results',default=str(ROOT))
 ap.add_argument('--outdir',default=str(OUT))
 a=ap.parse_args();ROOT=Path(a.results);OUT=Path(a.outdir);OUT.mkdir(parents=True,exist_ok=True)
 main()
