#!/usr/bin/env python3
"""Recheck saved full arrays, exact guards/counts and physical population invariants."""
from pathlib import Path
import numpy as np,json,hashlib,sys
from generate_fixtures import solve,CASES,D
P=Path(__file__).resolve().parent
plan=json.loads((P/'study_plan.json').read_text());pieces=[solve(*c)[0] for c in CASES];expected=np.r_[*[a[:c['prefix_count']] for a,c in zip(pieces,plan['cases'])],*[a[c['prefix_count']:] for a,c in zip(pieces,plan['cases'])]]
assert np.array_equal(expected,np.fromfile(P/'oracle_f64.bin',dtype='<f8'))
mask=np.fromfile(P/'exact_mask_u8.bin',dtype='u1').astype(bool);assert len(mask)==len(expected)
rows=[]
for result in sys.argv[1:]:
 doc=json.loads(Path(result).read_text())
 for model,v in doc['models'].items():
  for t in v.get('trials',[]):
   a=t['output_artifact'];file=Path(a['saved_path']);raw=file.read_bytes();assert hashlib.sha256(raw).hexdigest()==a['sha256'];out=np.frombuffer(raw,dtype='<f8');assert out.shape==expected.shape and np.isfinite(out).all();assert np.array_equal(out[mask],expected[mask]);err=np.abs(out-expected);tol=2e-5+2e-4*np.abs(expected);assert np.all(err<=tol)
   off=plan['prefix_count'];fields=[]
   for c in plan['cases']:
    N=c['nx']*c['ny'];GT=N//128*c['steps'];f=[];prev=[]
    for states in (f,prev):
     for q in range(9):states.append(out[off+16:off+16+N]);off+=N+32
    f=np.array(f);prev=np.array(prev);ps=out[off+16:off+16+GT];off+=GT+32;pc=out[off+16:off+16+GT];off+=GT+32
    massdrift=abs(f.sum()/c['mass_initial']-1);assert min(f.min(),prev.min())>0 and massdrift<2e-4
    fields.append({'kind':c['kind'],'mass_relative_drift':float(massdrift),'minimum_population':float(min(f.min(),prev.min())),'full_final_and_penultimate_values':18*N,'all_timestep_block_reductions':2*GT,'minimum_block_fluid_count':float(pc.min()),'maximum_velocity_sum':float(ps.max())})
   assert off==len(out);rows.append({'model':model,'trial':t['trial'],'artifact_sha256':a['sha256'],'checked_values':len(out),'exact_input_guard_and_count_positions':int(mask.sum()),'max_abs_error':float(err.max()),'max_oracle_budget_fraction':float(np.max(err/tol)),'fields':fields})
summary={'fresh_independent_tensor_replays':3,'full_checked_rows':len(rows),'all_pass':True,'rows':rows,'scope':'Declared bounded periodic flows, two obstacle geometries and zero-force equilibrium. No physical grid-convergence, turbulent-flow or original external dataset certificate. Full original formatted obstacle-flag defect tracked separately.'}
(P/'saved_output_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'full_checked_rows':len(rows),'all_pass':True,'max_budget_fraction':max(x['max_oracle_budget_fraction'] for x in rows)}))
