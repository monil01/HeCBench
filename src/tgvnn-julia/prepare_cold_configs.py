#!/usr/bin/env python3
"""Fresh no-reuse configs under a relocated source/asset root; no GPU invocation."""
import argparse,pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1];ORIGINAL='/home/imo/HeCBench_julia'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--assets',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--capture-root',type=pathlib.Path,required=True);z=a.parse_args();assets=z.assets.resolve();z.output.mkdir(parents=True,exist_ok=True);bindings=json.load(open(P/'final_source_binding_audit.json'))['source_bindings'];package={}
 for filename,digest in bindings.items():
  if '/.julia/packages/'not in filename:continue
  parts=filename.split('/.julia/packages/')[1].split('/');name=parts[0];suffix=pathlib.Path(*parts[2:]);candidates=[pathlib.Path(filename),*list((pathlib.Path.home()/'.julia/packages'/name).glob('*/'+str(suffix)))];f=next((f for f in candidates if f.is_file()and sha(f)==digest),None)
  if f is None:raise RuntimeError('Pinnedpackagefileunavailable/different: '+filename+'; instantiatepinnedprojectsfirst')
  package[filename]=str(f.resolve())
 def rewrite(x):
  if isinstance(x,str):
   if x in package:return package[x]
   x=x.replace('/tmp/tgvnn_oracle_v2_full500',str(assets/'pincat')).replace('/tmp/tgvnn_oracle_v2_rectangular',str(assets/'rectangular'));x=x.replace('/tmp/hecbench-portable-infrastructure-private/tgvnn',str(z.capture_root.resolve()/'private'));return x.replace(ORIGINAL,str(R))
  if isinstance(x,list):return[rewrite(v)for v in x]
  if isinstance(x,dict):return{k:rewrite(v)for k,v in x.items()}
  return x
 rows=[]
 for case in ['pincat','rectangular']:
  c=json.load(open(P/f'study_config_jacc_{case}.json'));c.pop('reuse_result',None);c.pop('reuse_models',None);c=rewrite(c)
  for model,s in c['models'].items():
   for name in s['source_dependencies']:assert pathlib.Path(name).is_file(),name
  out=z.output/f'{case}.json';out.write_text(json.dumps(c,indent=2)+'\n');rows.append({'case':case,'config':str(out),'sha256':sha(out),'all4fresh_processes_required':True})
 proof={'status':'prepared','GPU_execution':False,'source_root':str(R),'asset_root':str(assets),'configs':rows,'pinned_package_resolution':package,'source_manifest':str(P/'final_source_binding_audit.json'),'source_manifest_sha256':sha(P/'final_source_binding_audit.json'),'scope':'Newunmeasuredcoldconfigs, noreuse/fabricatedreceipt. Existingcanonical/tmp launchercontrolsSTATE separately; GPU6lockstillcentral.'};(z.output/'preparation_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':'prepared','configs':len(rows),'GPU_execution':False}))
if __name__=='__main__':main()
