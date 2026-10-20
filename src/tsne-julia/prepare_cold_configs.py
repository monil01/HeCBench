#!/usr/bin/env python3
"""Fresh no-reuse config from archived protocol, relocated CPU assets/source paths."""
import pathlib,json,argparse,hashlib,shutil,os
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1];ORIGINAL='/home/imo/HeCBench_julia'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--assets',type=pathlib.Path,required=True);a.add_argument('--native-build',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--pythonpath',required=True);a.add_argument('--julia',default=str(pathlib.Path.home()/'.juliaup/bin/julia'));z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);assets=z.assets.resolve();build=z.native_build.resolve();assert(build/'study_native').is_file()and(build/'bin/python3').is_file();assert all(not any(c.isspace()for c in str(x))for x in [assets,build]),'NativeCPUauditor requires whitespace-free oraclepath'
 original=json.load(open(P/'study_config_jacc_mnist_i1_metric_v4.json'));bindings=json.load(open(P/'final_saved_full_audit.json'))['strict_source_bindings'];packages={}
 for f,digest in bindings.items():
  if '/.julia/packages/'not in f:continue
  name,_,suffix=f.split('/.julia/packages/')[1].split('/',2);candidates=[pathlib.Path(f),*(pathlib.Path.home()/'.julia/packages'/name).glob('*/'+suffix)];actual=next((p for p in candidates if p.is_file()and sha(p)==digest),None);assert actual is not None,('Instantiatepinnedproject andmatchpackagefile',f);packages[f]=str(actual)
 def transform(v):
  if isinstance(v,str):
   if v in packages:return packages[v]
   if v=='/tmp/tsne_native_readiness/study_native_v2':return str(build/'study_native')
   if v.startswith('/tmp/tsne_native_readiness/source/'):
    old=v;new=R/'src/tsne-cuda'/pathlib.Path(v).name;assert sha(new)==bindings[old];return str(new)
   if v=='/home/imo/.juliaup/bin/julia':return z.julia
   return v.replace('/tmp/tsne_oracle_mnist_i1',str(assets/'i1')).replace(ORIGINAL,str(R))
  if isinstance(v,list):return [transform(x)for x in v]
  if isinstance(v,dict):return {k:transform(x)for k,x in v.items()}
  return v
 c=transform(original)
 for key in list(c):
  if key.startswith('reuse'):c.pop(key)
 c.update(configuration_id=original['configuration_id']+'_fresh_reproduction',configuration_role='diagnostic_reproduction',exclude_from_aggregate=True,exclusion_reason='Freshreproductionconfiguration; notautomaticreplacementofselectedimmutabletimingseries.')
 for model,spec in c['models'].items():
  spec['command']=['/usr/bin/env','PATH='+str(build/'bin')+':'+os.environ.get('PATH','/usr/bin:/bin'),'PYTHONPATH='+z.pythonpath,*spec['command']]
  spec['source_dependencies'] +=[str(build/'bin/python3'),str(build/'build_proof.json')]
  assert all(pathlib.Path(p).is_file()for p in spec['source_dependencies']),('missingdependency',model,[p for p in spec['source_dependencies']if not pathlib.Path(p).is_file()])
 out=z.output/'all4.json';out.write_text(json.dumps(c,indent=2)+'\n');proof={'status':'fresh_all4_config_prepared','config':str(out),'config_sha256':sha(out),'original_config_sha256':sha(P/'study_config_jacc_mnist_i1_metric_v4.json'),'all_declared_dependencies_exist':True,'immutable_references_reused':False,'CPU_audit_relocation_launcher':'Onlyknownnative/JuliaPythonauditpath/PYTHONPATH, process-localPATH; originalGPUandownedmeasuredsource unchanged.','GPU_execution':False,'source_root':str(R),'assets':str(assets),'native_build':str(build),'scope':'PreparationandCPUrebuildonly. No freshcoldGPUtrial or timingresult claimed.'};(z.output/'preparation_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
