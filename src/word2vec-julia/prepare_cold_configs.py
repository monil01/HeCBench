#!/usr/bin/env python3
"""Prepare a relocated, fresh all-four diagnostic configuration; no GPU."""
import pathlib,argparse,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--assets',type=pathlib.Path,required=True);a.add_argument('--native-build',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--julia',default=str(pathlib.Path.home()/'.juliaup/bin/julia'));z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);old=json.load(open(P/'study_config_jacc.json'));pkg={};bindings=json.load(open(P/'final_saved_full_audit.json'))['strict_source_bindings']
for path,digest in bindings.items():
 if '/.julia/packages/'not in path:continue
 name,_,suffix=path.split('/.julia/packages/')[1].split('/',2);candidates=[pathlib.Path(path),*(pathlib.Path.home()/'.julia/packages'/name).glob('*/'+suffix)];match=next((p for p in candidates if p.is_file()and sha(p)==digest),None);assert match is not None,('Instantiatepinnedproject',path);pkg[path]=str(match)
def tr(v):
 if isinstance(v,str):
  if v in pkg:return pkg[v]
  if v=='/tmp/word2vec_native_readiness/study_native':return str(z.native_build.resolve()/'study_native')
  if v=='/home/imo/.juliaup/bin/julia':return z.julia
  return v.replace('/home/imo/HeCBench_julia',str(R)).replace('/tmp/word2vec_assets_v2',str(z.assets.resolve()))
 if isinstance(v,list):return [tr(x)for x in v]
 if isinstance(v,dict):return {k:tr(x)for k,x in v.items()}
 return v
c=tr(old)
for k in list(c):
 if k.startswith('reuse'):c.pop(k)
c.update(configuration_id=c['configuration_id']+'_fresh_reproduction',configuration_role='diagnostic_reproduction',exclude_from_aggregate=True,exclusion_reason='Freshreproductiondoesnotreplaceselectedimmutabletimingseries.')
for model,spec in c['models'].items():
 assert all(pathlib.Path(p).is_file()for p in spec['source_dependencies']),(model,'missingdeclaredfile');assert pathlib.Path(spec['command'][0]).is_file()or spec['command'][0]=='python3'
f=z.output/'all4.json';f.write_text(json.dumps(c,indent=2)+'\n');proof=dict(status='fresh_all4_config_prepared',config=str(f),config_sha256=sha(f),source_root=str(R),all_declared_files_exist=True,GPU_execution=False,reuse_of_old_trials=False,scope='CPUconfigurationpreparationonly, noadditionalfiveGPUseries/freshGPUreproductionclaim.');(z.output/'preparation_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
