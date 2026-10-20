#!/usr/bin/env python3
"""Prepare fresh all-four configurations in a relocated source copy; no GPU launch."""
import pathlib,json,argparse,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1];ORIGINAL_ROOT='/home/imo/HeCBench_julia'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--capture-root',type=pathlib.Path,required=True);z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);proof=json.load(open(P/'final_source_binding_audit.json'));bindings=proof['source_bindings'];package={}
 for f,h in bindings.items():
  if '/.julia/packages/'not in f:continue
  part=f.split('/.julia/packages/')[1].split('/');pkg=part[0];rel=pathlib.Path(*part[2:]);candidates=[pathlib.Path(f),*list((pathlib.Path.home()/'.julia/packages'/pkg).glob('*/'+str(rel)))];match=next((p for p in candidates if p.is_file()and sha(p)==h),None)
  if match is None:raise RuntimeError('Pinnedinstalledpackagefileunavailable/hashmismatch: '+f+';instantiatepinnedenvironmentfirst')
  package[f]=str(match.resolve())
 def rewrite(x):
  if isinstance(x,str):
   if x in package:return package[x]
   return x.replace(ORIGINAL_ROOT,str(R))
  if isinstance(x,list):return [rewrite(y)for y in x]
  if isinstance(x,dict):return {k:rewrite(y)for k,y in x.items()}
  return x
 out=[]
 for case in ['NYR','NYR_gpuonly','weighted_control','narrow_chain']:
  c=rewrite(json.load(open(P/f'study_config_jacc_{case}.json')))
  for key in ['reuse_result','reuse_models','storage_policy']:c.pop(key,None)
  for model,m in c['models'].items():m['command'][-1]=str(z.capture_root.resolve()/'private'/case/model)
  dest=z.output/(case+'.json');dest.write_text(json.dumps(c,indent=2)+'\n');out.append(str(dest.resolve()))
 print(json.dumps(dict(configs=out,source_root=str(R),GPU_launched=False,all4_fresh=True,reuse_of_historical_measured_trials=False,package_hash_matches=package),indent=2))
if __name__=='__main__':main()
