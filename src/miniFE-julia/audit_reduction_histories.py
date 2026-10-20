#!/usr/bin/env python3
"""Rehash and compare every retained post-final-reduction buffer; no GPU calls."""
import argparse,pathlib,json,hashlib,math,struct
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();r=json.loads(a.result.read_text());cfg=json.loads((a.result.parent/'config.json').read_text());atol=cfg['atol'];rtol=cfg['rtol'];cache={};rows=[];ok=True
for model,m in r['models'].items():
 for t in m.get('trials',[]):
  h=t['dot_history_artifact'];f=pathlib.Path(h['path']);b=f.read_bytes();v=struct.unpack('<'+'d'*(len(b)//8),b);sha=hashlib.sha256(b).hexdigest();valid=len(v)==h['count'] and len(v)%257==0 and all(map(math.isfinite,v)) and all(v[i] in (2,16,128)for i in range(0,len(v),257));cache[model,t['trial']]=(b,v,sha);rows.append({'model':model,'trial':t['trial'],'path':str(f),'bytes':len(b),'sha256':sha,'records':len(v)//257,'full_finite_shape_pass':valid});ok &= valid
comparisons=[]
for row in rows:
 for reference in ('cuda','cuda_jl'):
  key=(reference,row['trial'])
  if key not in cache:continue
  b,v,sha=cache[row['model'],row['trial']];rb,rv,rsha=cache[key];same_shape=len(v)==len(rv);good=same_shape;mx=mr=0.0
  if same_shape:
   for i,(x,y)in enumerate(zip(v,rv)):
    delta=abs(x-y);mx=max(mx,delta);mr=max(mr,delta/max(abs(x),abs(y),1e-30))
    good &= (x==y if i%257==0 else delta<=atol+rtol*abs(y))
  comparisons.append({'model':row['model'],'reference':reference,'trial':row['trial'],'count':len(v),'source_sha256':sha,'reference_sha256':rsha,'byte_identical':b==rb,'max_abs_error':mx,'max_rel_error_symmetric_denominator_floor1e_minus30':mr,'pass':good});ok &=good
out={'benchmark':'miniFE','result_path':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'atol':atol,'rtol':rtol,'scope':'Complete actual 256-element device reduction state after every final-reduce/copy. Record dimension must agree exactly; floats use predeclared artifact tolerance. Warmup sidecars are retained but not counted among five numerical trial comparisons. No GPU rerun.','artifacts':rows,'comparisons':comparisons,'all_pass':bool(ok)};a.output.parent.mkdir(parents=True,exist_ok=True);q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps({'all_pass':bool(ok),'artifacts':len(rows),'comparisons':len(comparisons)}));raise SystemExit(0 if ok else 2)
