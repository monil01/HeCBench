"""Stricter saved-series fieldwise dual checks; all source/run/time records remain unchanged."""
from pathlib import Path
import json,hashlib,struct
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=R/'.porting-state/portable_julia/runs/20261019T170239306493Z_braycurtis/result.json';r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());records=[]
for model,reference in [('cuda_jl','cuda'),('ka','cuda'),('ka','cuda_jl'),('jacc','cuda'),('jacc','cuda_jl')]:
 for t,u in zip(r['models'][model]['trials'],r['models'][reference]['trials']):
  arrays=[];raws=[];proofs=[]
  for row in [t,u]:
   a=row['output_artifact'];p=Path(a['saved_path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256'];raws.append(b);arrays.append(struct.unpack('<'+'d'*plan['artifact_count'],b));proofs.append(dict(path=str(p.relative_to(R)),sha256=a['sha256']))
  x,y=arrays;off=plan['input_prefix_count'];assert raws[0][:off*8]==raws[1][:off*8];summaries=[]
  for c in plan['cases']:
   at,rt=(1e-5,1e-4)if c['dtype']==32 else(1e-12,1e-10)
   for name,count in zip(['distance','absdiff','absplus','gradx1','gradx2'],[c['pairs']]*3+[c['x1_count'],c['x2_count']]):
    a,b=x[off:off+count+16],y[off:off+count+16];off+=count+16;assert a[:8]==b[:8]and a[-8:]==b[-8:];error=[abs(z-w)for z,w in zip(a[8:-8],b[8:-8])];budget=[at+rt*abs(w)for w in b[8:-8]];assert all(d<=v for d,v in zip(error,budget));summaries.append(dict(case=c['id'],dtype=c['dtype'],field=name,count=count,atol=at,rtol=rt,max_abs_error=max(error),max_budget_fraction=max(d/v for d,v in zip(error,budget))))
  assert off==len(x);records.append(dict(model=model,reference=reference,trial=t['trial'],saved_artifacts_rehashed=proofs,exact_input_prefix_bytes=True,fieldwise_metrics=summaries))
(P/'per_field_dual_reference_audit.json').write_text(json.dumps(dict(result_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),scope='AdditionalCPU-onlystrictfieldwisecomparisonofall25savedtrialpairs underalreadydeclaredindependentF32/F64fieldbudgets; originalglobalrunnercontract/results unchanged.',records=records,all25_pass=True,no_new_GPU_execution=True),indent=2)+'\n');print('BRAY_STRICT_FIELDWISE_DUAL_ALL25_PASS')
