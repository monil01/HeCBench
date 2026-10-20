#!/usr/bin/env python3
"""Post-hoc descriptive physical-field errors vs saved native CUDA, not new gates."""
import argparse,pathlib,json,array,sys,hashlib
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();r=json.loads(a.result.read_text());native=r['models']['cuda']['trials'];out={}
def load(t):
 p=pathlib.Path(t['output_artifact']['saved_path']);raw=p.read_bytes();v=array.array('d');v.frombytes(raw)
 if sys.byteorder!='little':v.byteswap()
 return v,dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
for model in ['cuda_jl','ka','jacc']:
 metrics={};pairs=[]
 def add(name,x,y):
  z=metrics.setdefault(name,dict(max_abs_error=0.,checked_values=0));z['max_abs_error']=max(z['max_abs_error'],abs(x-y));z['checked_values']+=1
 for nt,tt in zip(native,r['models'][model]['trials']):
  x,xev=load(nt);y,yev=load(tt);assert len(x)==len(y);off=47220
  while off<len(x):
   assert x[off]==y[off];tag=x[off];off+=1
   if tag==-16001:
    n=int(x[off]);assert n==int(y[off])==3371;off+=26
    for i in range(n):
     for c in range(3):add('force_input_position_Angstrom',x[off+14*i+c],y[off+14*i+c])
    off+=14*n
    for i in range(n):
     for c in range(3):add('returned_force_K_per_Angstrom',x[off+14*i+6+c],y[off+14*i+6+c])
    off+=14*n
   elif tag==-17001:
    nm=int(x[off+2]);assert nm==int(y[off+2])==955
    for j,k in [(4,'thermostat_multiplier_inverse_fs'),(6,'last_computed_potential_K'),(7,'last_computed_kinetic_K'),(8,'last_computed_total_energy_K'),(9,'last_computed_temperature_K')]:add(k,x[off+j],y[off+j])
    off+=10
    for mi in range(nm):
     na=int(x[off]);assert na==int(y[off]);
     for group,name in [(0,'molecular_force_K_per_Angstrom'),(2,'molecular_COM_Angstrom'),(6,'molecular_velocity_Angstrom_per_fs')]:
      for c in range(3):add(name,x[off+5+3*group+c],y[off+5+3*group+c])
     off+=41
     for i in range(na):
      for c in range(3):add('host_atom_position_Angstrom',x[off+27*i+6+c],y[off+27*i+6+c])
     off+=27*na
   else:raise AssertionError((off,tag))
  assert off==len(x);pairs.append(dict(trial=tt['trial'],native=xev,target=yev))
 out[model]=dict(metrics=metrics,source_pairs=pairs)
result=dict(benchmark='mcmd',analysis_kind='Post-hoc descriptive per-field maximum absolute differences against native CUDA; original acceptance unchanged.',result_path=str(a.result),result_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),models=out,notes=['Statistics reflect original CPU update cadence; intermediate snapshots repeat last computed statistics.','Molecular velocity is the active translation variable for this molecular-mode source; not an atomic-mode validation.','Both GPU reference comparisons and the independent long-double force gate are separately preserved.'])
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:v['metrics']for m,v in out.items()},indent=2))
