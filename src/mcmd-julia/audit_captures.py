#!/usr/bin/env python3
"""Full saved-artifact rehash, schema, initial-bit, state and rigid-body replay.
The separately recorded runtime long-double force gate is not relabeled as a
second independently recomputed force calculation by this structural replay.
"""
import argparse,json,pathlib,hashlib,math,array,sys
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();res=json.loads(a.result.read_text());rows=[];canonical=None
for model,m in res['models'].items():
 if not m.get('complete'):continue
 for trial in m['trials']:
  f=pathlib.Path(trial['output_artifact']['saved_path']);raw=f.read_bytes();x=array.array('d');x.frombytes(raw);
  if sys.byteorder!='little':x.byteswap()
  n=int(x[0]);prefix=26+14*n;assert n==3371 and prefix==47220
  if canonical is None:canonical=raw[:prefix*8]
  assert raw[:prefix*8]==canonical and all(map(math.isfinite,x));off=prefix;force=state=0;rigids=None;state_times=[];maxrigid=0
  while off<len(x):
   tag=x[off];off+=1
   if tag==-16001:
    h=x[off:off+26];off+=26;nn,block,pf=int(h[0]),int(h[1]),int(h[2]);assert nn==n and block==256 and pf==1 and int(h[5])==3 and int(h[6])==1
    before_off=off;off+=14*n;after_off=off;off+=14*n
    for atom in range(n):
     bo=8*(before_off+14*atom);ao=8*(after_off+14*atom);assert raw[bo:bo+48]==raw[ao:ao+48] and raw[bo+72:bo+112]==raw[ao+72:ao+112];assert all(x[before_off+14*atom+c]==0 for c in range(6,9))
    force+=1
   elif tag==-17001:
    step,time,nm,nn,lm,q,*obs=x[off:off+10];off+=10;assert int(nn)==n and int(nm)==955 and step==state and time==state;state_times.append(float(time));current=[];atoms=0
    for mi in range(int(nm)):
     meta=x[off:off+41];off+=41;na=int(meta[0]);assert na>0 and meta[1]>0
     data=x[off:off+27*na];off+=27*na;atoms+=na;pos=[data[27*i+6:27*i+9]for i in range(na)];rr=[]
     for i in range(na):
      for j in range(i+1,na):rr.append(sum((pos[i][c]-pos[j][c])**2 for c in range(3)))
     current.append(rr)
     if rigids is not None:
      err=[abs(v-w)for v,w in zip(rr,rigids[mi])];assert all(e<=1e-7+1e-10*w for e,w in zip(err,rigids[mi]));maxrigid=max(maxrigid,max(err,default=0))
    assert atoms==n
    if rigids is None:rigids=current
    state+=1
   else:raise AssertionError(('Unknown full-state marker',off-1,tag))
  assert off==len(x) and force==state==11
  rows.append(dict(model=model,trial=trial['trial'],path=str(f),sha256=hashlib.sha256(raw).hexdigest(),values=len(x),actual_input_prefix_values=prefix,force_states=force,full_molecular_states=state,max_rigid_squared_distance_error=maxrigid,all_finite=True,all_nonforce_atom_bytes_unchanged=True,initial_input_bits_exact=True,independent_force_gate='Runtimeallcoordinateslongdouble; not independentlyrecomputedbythisstructuralparser'))
out=dict(benchmark='mcmd',full_rehash_count=len(rows),all_pass=True,scope='Fullartifactparse, allactualreadonlyatomfields, all11molecularstates/rigidpairs; runtimeindependentforceoracleseparate',rows=rows);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
