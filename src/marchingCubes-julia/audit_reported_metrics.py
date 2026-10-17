#!/usr/bin/env python3
"""Recompute CPU error metrics from immutable saved arrays; no GPU execution."""
from pathlib import Path
import array,datetime,hashlib,json,sys
P=Path(__file__).resolve().parent;R=P.parent.parent;run=R/'.porting-state/portable_julia/runs/20261017T211206803388Z_marchingCubes/result.json';r=json.loads(run.read_text());expected=array.array('d');expected.frombytes((P/'study_expected_f64.bin').read_bytes());rows=[]
for model,x in r['models'].items():
 for q in x['trials']:
  f=Path(q['output_artifact']['saved_path']);data=f.read_bytes();got=array.array('d');got.frombytes(data);assert len(got)==len(expected)==401053
  # Prefix is exactly checked elsewhere; these metrics cover actual numeric outputs.
  absmax=relmax=0.
  for a,b in zip(got[4401:],expected[4401:]):
   error=abs(a-b);absmax=max(absmax,error);relmax=max(relmax,error/max(abs(b),.001))
  rows.append({'model':model,'trial':q['trial'],'true_CPU_max_abs_error':absmax,'true_CPU_max_relative_error_with_atol_floor':relmax,'raw_reported_max_rel_error':q['max_rel_error'],'artifact_sha256':hashlib.sha256(data).hexdigest(),'saved_artifact':str(f)})
erratum={'field':'models[cuda_jl|ka|jacc].trials[*].max_rel_error','raw_definition':'max_abs_error / atol, scaled absolute residual; not per-element mathematical relative error.','native_raw_definition':'maximum abs(got-expected)/max(abs(expected),atol,1e-300).','central_dual_metric_definition':'Independently computed by verifier; unaffected.','corrected_CPU_definition':'Maximum abs(got-expected)/max(abs(expected),0.001), excluding exact input-prefix. Mixed preprocessing/property artifact; not a full-mesh geometric error.','acceptance_and_execution':'Unchanged absolute-only tolerance0.001 and exact prefix checks. No timing/source/input/output/acceptance mutation or retiming.','independent_recomputed_metrics':rows,'oracle_file_sha256':hashlib.sha256((P/'study_expected_f64.bin').read_bytes()).hexdigest(),'auditor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'metric_definition_erratum.json').write_text(json.dumps(erratum,indent=2)+'\n');entry={'run_id':r['run_id'],'classification_only':True,'result_sha256':hashlib.sha256(run.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((run.parent/'config.json').read_bytes()).hexdigest(),'reason':'Clarify raw customJulia metric label and provide independently recomputed CPU error metrics from saved outputs; no source/run/acceptance/timing changes.','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'updates':{'reported_metric_definition_erratum':erratum}}
side=P/'study_adjudications.json';doc=json.loads(side.read_text())if side.exists()else{'run_adjudications':[]};old=next((e for e in doc['run_adjudications']if e['run_id']==r['run_id']),None)
if old:
 assert old['result_sha256']==entry['result_sha256']and old['config_sha256']==entry['config_sha256'];old['updates']['reported_metric_definition_erratum']=erratum;old['reason']+='; metric definition erratum added'
else:doc['run_adjudications'].append(entry)
side.write_text(json.dumps(doc,indent=2)+'\n');print({m:max(q['true_CPU_max_relative_error_with_atol_floor']for q in rows if q['model']==m)for m in r['models']})
