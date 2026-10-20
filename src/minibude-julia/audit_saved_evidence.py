#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib,sys
P=Path(__file__).resolve().parent;R=P.parents[1];rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());c=json.loads((P/'asset_contract.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[];rawfiles=0;rawbytes=0;maxerr=0.;maxratio=0.;sources=0
for model,record in r['models'].items():
 if not record.get('complete'):continue
 for f,h in record['adapter_file_hashes'].items():assert sha(Path(f))==h,(model,f);sources+=1
 for trial in record['trials']:
  art=trial['output_artifact'];path=Path(art['saved_path']);assert sha(path)==art['sha256'];v=np.memmap(path,dtype='<f8',mode='r');assert len(v)==c['artifact_count'];at=0;tails=[];hashes={};base=trial['raw_state_prefix'];trialmax=0.;trialratio=0.
  for case in c['cases']:
   k=case['index'];n=case['n'];np_=case['protein'];nl=case['ligand'];nf=case['ntypes'];header=[k,n,np_,nl,nf,7,256,(n+255)//256,1];assert np.array_equal(v[at:at+9],header);at+=9;inputs={};expected={};poses=np.fromfile(P/f'case{k}_poses.bin',dtype='<u4').reshape(6,n)
   for key,size in [('protein',16),('ligand',16),('forcefield',16)]:
    file=P/f'case{k}_{key}.bin';expected[key]=b'\xa5'*(16*size)+file.read_bytes()+b'\xa5'*(16*size)
   for j in range(6):expected[f'pose{j}']=b'\xa5'*64+poses[j].tobytes()+b'\xa5'*64
   expected['energy']=b'\xa5'*64+np.full(n,-777,dtype='<f4').tobytes()+b'\xa5'*64
   order=['protein','ligand','forcefield']+[f'pose{j}'for j in range(6)]+['energy'];arrays={}
   for key in order:
    arrays[key]={}
    for phase in ['before','after']:
     f=Path(f'{base}_case{k}_{key}_{phase}.bin');b=f.read_bytes();assert len(b)==len(expected[key]);hashes[str(f.relative_to(R))]=sha(f);rawfiles+=1;rawbytes+=len(b);a=np.frombuffer(b,dtype='<u4');arrays[key][phase]=a
     assert b[:64]==b'\xa5'*64 and b[-64:]==b'\xa5'*64
     if phase=='before'or key!='energy':assert b==expected[key],(model,trial['trial'],k,key,phase)
    a=arrays[key]['before'];assert np.array_equal(v[at:at+len(a)],a.astype('f8'));at+=len(a)
   for key in order[:-1]:
    a=arrays[key]['after'];assert np.array_equal(v[at:at+len(a)],a.astype('f8'));at+=len(a)
   out=arrays['energy']['after'];e=out.view('<f4')[16:-16].astype('f8');oracle=np.fromfile(P/f'case{k}_oracle_channels_f64.bin',dtype='<f8').reshape(n,6)[:,0];error=abs(e-oracle);budget=.05+.0002*abs(oracle);assert np.isfinite(e).all()and(error<=budget).all();trialmax=max(trialmax,float(error.max()));trialratio=max(trialratio,float((error/budget).max()));tails.append((e,out[:16],out[-16:]))
  assert at==c['input_prefix_count']
  for e,lo,hi in tails:
   assert np.array_equal(v[at:at+len(e)],e);at+=len(e);assert np.array_equal(v[at:at+16],lo.astype('f8'));at+=16;assert np.array_equal(v[at:at+16],hi.astype('f8'));at+=16;assert v[at]==1;at+=1
  assert at==len(v);maxerr=max(maxerr,trialmax);maxratio=max(maxratio,trialratio);rows.append({'model':model,'trial':trial['trial'],'artifact_path':str(path.relative_to(R)),'artifact_sha256':sha(path),'raw_file_hashes':hashes,'maximum_independent_F64_energy_error':trialmax,'maximum_error_over_declared_budget':trialratio,'all_actual_inputs_initial_outputs_readonly_bytes_and_guards_exact':True})
receipt={'all_pass':True,'result':str(rp.relative_to(R)),'result_sha256':sha(rp),'trials':len(rows),'raw_files':rawfiles,'raw_bytes':rawbytes,'source_dependency_bindings_rehashed':sources,'full_pose_energies':sum(x['n']for x in c['cases'])*len(rows),'maximum_independent_F64_energy_error':maxerr,'maximum_error_over_declared_budget':maxratio,'rows':rows,'oracle_independence_audit':'src/minibude-julia/independent_oracle_audit.json','input_prefix_encoding':'ExactrawUInt32words encodedasFloat64; structABI16B/fieldoffsets0,4,8,12','scope':'Declared globalstates; no inference of rawshared/private transient register values.'};(P/f'raw_state_audit_{len(rows)}trials.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='rows'}))
