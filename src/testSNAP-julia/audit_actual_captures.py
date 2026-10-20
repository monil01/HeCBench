#!/usr/bin/env python3
"""CPU-only full actual input/guard/every-step scientific-output audit; requires NumPy."""
import argparse,json,gzip,hashlib,struct
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();source=a.source;root=source.parents[1]/'.porting-state/portable_julia/testSNAP_primary';res=json.loads(a.result.read_text());steps=[100,3,3];cases=[];expectedprefix=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for ci,J in enumerate([14,4,2]):
 b=(source/f'data/J{J}_source_fixture.bin').read_bytes();h=struct.unpack_from('<12I',b);_,J,na,ng,nb,nco,nu,nd,nz,ncg,jdim,pdim=h;sc=np.frombuffer(b,dtype='<f8',count=4,offset=48);at=80
 sizes=[jdim,nu,jdim,9*nz,jdim**3,pdim**2,ncg,nz,na*nb*3,na*nb,na*nb,na*nb,3*(na+ng),nco+1];types=['<i4']*5+['<f8']*4+['<i4']+['<f8']*4;blocks=[]
 for n,t in zip(sizes,types):v=np.frombuffer(b,dtype=t,count=n,offset=at);blocks.append(v);at+=n*np.dtype(t).itemsize
 assert at==len(b);mapping={i:blocks[i]for i in range(9)};mapping.update({9:blocks[10],10:blocks[11],11:blocks[9].astype('<f8'),12:blocks[12],13:blocks[13]})
 expectedprefix.extend([np.asarray(h,dtype='<f8'),sc,np.asarray([steps[ci]],dtype='<f8')]);expectedprefix.extend(mapping[i].astype('<f8')for i in list(range(8))+[12,13,8,9,10,11]);oracle=np.fromfile(source/f'data/J{J}_independent_oracle.bin',dtype='<f8');assert len(oracle)==3*na*nb+3*(na+ng);cases.append({'h':h,'mapping':mapping,'oracle':oracle,'nd':3*na*nb,'nf':3*(na+ng),'ref':mapping[12]})
expectedprefix=np.concatenate(expectedprefix).astype('<f8');assert len(expectedprefix)==632233
expectedrows={(ci,step,i)for ci,N in enumerate(steps)for step in range(1,N+1)for i in range(8,12)}|{(ci,0,i)for ci in range(3)for i in list(range(8))+[12,13]}|{(ci,N+1,i)for ci,N in enumerate(steps)for i in range(8)}
unique={};ledgers=[];groups={};guardbytes=0
for f in sorted((root/a.model).glob('*_trial[1-5].inputs.jsonl')):
 nonce,trial=f.name.rsplit('_trial',1);trial=int(trial.split('.')[0]);groups.setdefault(nonce,set()).add(trial);rows=[json.loads(l)for l in f.read_text().splitlines()];assert len(rows)==len(expectedrows)==478;assert{(r['case'],r['step'],r['id'])for r in rows}==expectedrows
 for row in rows:
  ci,i=row['case'],row['id'];s=row['sha256'];n=row['bytes'];assert row['device_input']==(i<=10);x=cases[ci]['mapping'][i].tobytes()
  if s not in unique:
   raw=gzip.decompress((root/'CAS'/(s+'.bin.gz')).read_bytes());assert hashlib.sha256(raw).hexdigest()==s;unique[s]=raw
  assert len(unique[s])==n and unique[s]==x
 gf=f.with_name(f.name.replace('.inputs.jsonl','.guards.bin'));raw=gf.read_bytes();assert len(raw)==16*sum(steps)*528;seen=set()
 for off in range(0,len(raw),528):
  ci,step,i,n=struct.unpack_from('<4i',raw,off);assert 0<=ci<3 and 1<=step<=steps[ci]and 0<=i<16 and n==512;assert(ci,step,i)not in seen;seen.add((ci,step,i));assert raw[off+16:off+528]==b'\xa5'*512;guardbytes+=512
 assert len(seen)==16*sum(steps);ledgers.append({'path':str(f),'sha256':sha(f),'guard_path':str(gf),'guard_sha256':hashlib.sha256(raw).hexdigest()})
assert len(groups)==1 and next(iter(groups.values()))=={1,2,3,4,5}
model=res['models'][a.model];errors=[];artifactrecords=[];total=0;maximum=0.;normmax=0.
for row in model['trials']:
 ar=row['output_artifact'];f=Path(ar['saved_path']);assert sha(f)==ar['sha256'];v=np.memmap(f,dtype='<f8',mode='r');assert len(v)==17715669 and v[:len(expectedprefix)].tobytes()==expectedprefix.tobytes();pos=len(expectedprefix)
 for ci,c in enumerate(cases):
  sse=0.;N=steps[ci];nd,nf=c['nd'],c['nf']
  for step in range(1,N+1):
   assert tuple(v[pos:pos+2])==(ci,step);pos+=2;actual=v[pos:pos+nd+nf];pos+=nd+nf;assert np.all(np.isfinite(actual));e=np.abs(actual-c['oracle']);ratios=e/(1e-7+1e-9*np.abs(c['oracle']));maximum=max(maximum,float(e.max()));normmax=max(normmax,float(ratios.max()));assert np.all(ratios<=1);force=actual[nd:];delta=force-c['ref']
   if ci!=1:assert np.all(np.abs(delta)<=1e-7+1e-9*np.abs(c['ref']))
   sse+=float(np.sum(delta*delta));total+=nd+nf
  actualsse,rms=v[pos:pos+2];pos+=2;assert abs(actualsse-sse)<=1e-18+1e-12*abs(sse);assert abs(rms-np.sqrt(actualsse/(nf//3*N)))<=1e-15+1e-12*abs(rms)
 assert pos==len(v);artifactrecords.append({'trial':row['trial'],'path':str(f),'sha256':ar['sha256']})
out={'model':a.model,'result':str(a.result),'result_sha256':sha(a.result),'actual_input_prefix_values':len(expectedprefix),'input_ledgers_all5':ledgers,'input_rows_per_trial':478,'complete_input_CAS_frames_rehashed':len(unique),'unique_CAS_bytes_rehashed':sum(map(len,unique.values())),'actual_guard_bytes_all5':guardbytes,'all106timedstep_outputs_each_trial':True,'pointwise_physical_gradient_force_comparisons_all5':total,'max_abs_vs_independent_oracle':maximum,'max_normalized_independent_budget':normmax,'J4_supplied_zero_reference_used_for_acceptance':False,'J14_J2_full_provided_reference_checks':True,'artifacts':artifactrecords,'issues':errors,'scope':'AllactualstaticGPUinputssetup/final, everytimedstep3H2Dinputs+hostneighborindicesfullbitchecked/retainedCAS; everyactualGPUallocationguardbytesretained. Completeevery-stepdedr/forces, notallhugeinternalU/dU/Yworkspacevalues. Independentoraclev3CGpolynomial/dual/energytensor; nohistoricalreconstruction.'}
a.receipt.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in['input_ledgers_all5','artifacts']}))
