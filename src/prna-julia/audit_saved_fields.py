#!/usr/bin/env python3
"""Complete PRNA artifact decoding, strict field budgets, actual inputs and controls.
NumPy is a CPU analysis dependency only; records its actual version. No GPU work.
"""
import argparse,json,hashlib,itertools,struct
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
result=json.loads(args.result.read_text());plan=json.loads((P/'study_plan.json').read_text());ns=[x['n']for x in plan['cases']];params=np.fromfile(P/'params_u32.bin',dtype='<u4');seq=np.fromfile(P/'sequences_i32.bin',dtype='<i4');bcp=np.fromfile(P/'base_can_pair_i32.bin',dtype='<i4');gold=np.fromfile(P/'CPU_final_states_f32.bin',dtype='<f4');tracegold=np.fromfile(P/'CPU_trace27_allstages_f32.bin',dtype='<f4');tiny=np.fromfile(P/'tiny_oracle_f64.bin',dtype='<f8')
guardF=np.float32(-987654.25);guardbits=np.array([guardF],dtype='<f4').view('<u4')[0];infbit=np.uint32(0x7f800000)
def sizes(n):return [('v',n*n),('w5',n+1),('w3',n+1),('z',n*n),('yl',n*n),('y',n*n),('wq',n*(n-1)//2),('w',2*n),('wl',2*n),('xl',2*n),('x',5*n)]
def with_guard(bits,g):return np.concatenate([np.full(17,g,dtype='<u4'),np.asarray(bits,dtype='<u4'),np.full(17,g,dtype='<u4')])
expectedprefix=[with_guard(params,np.uint32(0xa5a5a5a5))];sequences=[];masks=[];sa=1;ba=0
for n in ns:
 assert seq[sa]==n;sa+=1;s=seq[sa:sa+n];sa+=n;nb=n*(n-1)//2;mask=bcp[ba:ba+nb];ba+=nb;sequences.append(s);masks.append(mask)
for s,mask,n in list(zip(sequences,masks,ns))+[(sequences[-1],masks[-1],27)]:
 expectedprefix.extend([with_guard(s.view('<u4'),np.uint32(0xa5a5a5a5)),with_guard(mask.view('<u4'),np.uint32(0xa5a5a5a5))])
 expectedprefix.extend(with_guard(np.full(k,infbit,dtype='<u4'),guardbits)for _,k in sizes(n))
prefix=np.concatenate(expectedprefix).astype('<f8');assert len(prefix)==935631
reports=[];decoded={};fieldschema=None
for model,data in result['models'].items():
 if data.get('status')!='native_check_pass':continue
 assert len(data['trials'])==5
 for row in data['trials']:
  item=row['output_artifact'];p=Path(item['saved_path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'];a=np.frombuffer(raw,dtype='<f8');assert len(a)==4880172;assert a[:len(prefix)].tobytes()==prefix.tobytes()
  t=a[len(prefix):].reshape(-1,3);numeric,cls,scaled=t.T;u64=np.ldexp(scaled,64);assert np.all(np.isfinite(u64))and np.all((0<=u64)&(u64<=4294967295))and np.all(u64==np.floor(u64));bits=u64.astype('<u4');vals=bits.view('<f4')
  expectedcls=np.where(np.isfinite(vals),0,np.where(np.isnan(vals),2,np.where(vals>0,1,-1)))
  assert np.array_equal(cls,expectedcls)and not np.any(np.isnan(vals));assert np.array_equal(numeric,np.where(np.isfinite(vals),vals.astype('<f8'),0.0))
  at=0;ga=0;schema=[];maxstate=0.0;probmax=[]
  def buffer(name,k,expected):
   global at,maxstate
   start=at;z=vals[at:at+k+34];ub=bits[at:at+k+34];assert len(z)==k+34;assert np.all(ub[:17]==guardbits)and np.all(ub[-17:]==guardbits);x=z[17:-17];assert np.array_equal(np.isposinf(x),np.isposinf(expected));assert not np.any(np.isneginf(x));finite=np.isfinite(expected);er=np.abs(x[finite].astype('f8')-expected[finite].astype('f8'));assert np.all(er<=.002+2e-5*np.abs(expected[finite].astype('f8')));maxstate=max(maxstate,float(er.max(initial=0)));schema.append((start+17,start+17+k,'state'));at+=k+34;return x
  for q,n in enumerate(ns):
   current={}
   for name,k in sizes(n):current[name]=buffer(name,k,gold[ga:ga+k]);ga+=k
   start=at;pr=vals[at:at+n*n];assert len(pr)==n*n;assert np.all(np.isfinite(pr))and np.all((pr>=0)&(pr<=1.002));matrix=pr.reshape(n,n);assert np.array_equal(matrix,matrix.T);assert float(matrix.astype('f8').sum(axis=1).max())<=1.002
   mask=np.zeros((n,n),dtype=bool)
   for j in range(n):
    for i in range(j):mask[i,j]=mask[j,i]=masks[q][j*(j-1)//2+i]!=0
   assert np.all(matrix[~mask]==0)
   energy=(current['v'].reshape(n,n)+current['v'].reshape(n,n).T-current['w3'][0]).astype('<f4');reference=np.zeros((n,n),dtype='f8');reference[mask]=np.exp(-energy[mask].astype('f8'));assert np.all(np.abs(matrix.astype('f8')-reference)<=.0002+.0002*np.abs(reference))
   if q==1:
    assert abs(float(current['w3'][0])-tiny[0])<2e-6 and abs(float(matrix[0,6])-tiny[1])<2e-6 and abs(float(matrix[1,5])-tiny[2])<2e-6
   if q==2:assert current['w3'][0]==0 and np.all(matrix==0)
   if q==3:assert abs(float(current['w3'][0])-tiny[3])<2e-6 and abs(float(matrix[0,6])-tiny[4])<2e-6
   schema.append((at,at+n*n,'probability'));at+=n*n;probmax.append(float(matrix.max()))
  assert ga==len(gold);ga=0
  for snap in range(209):
   for name,k in sizes(27):buffer('trace27_'+str(snap)+'_'+name,k,tracegold[ga:ga+k]);ga+=k
  assert ga==len(tracegold)and at==len(vals)
  if fieldschema is None:fieldschema=schema
  else:assert schema==fieldschema
  decoded[model,row['trial']]=(vals.copy(),bits.copy())
  reports.append({'model':model,'trial':row['trial'],'path':str(p),'sha256':item['sha256'],'bytes':len(raw),'all_input_bytes_exact':7485048,'full_output_raw_Float32_values_decoded':len(vals),'all_NaNs_rejected':True,'all_structural_Inf_classes_exact_source_CPU':True,'all_guards_exact':True,'all_finite_channels_exact_raw_decode':True,'source_CPU_state_max_abs_error':maxstate,'tiny_partition_controls':'pass','probability_max_by_case':probmax,'untimed_trace27_snapshots':209})
pairs=[]
models=sorted({m for m,t in decoded})
for left,right in itertools.combinations(models,2):
 for trial in range(1,6):
  a,ab=decoded[left,trial];b,bb=decoded[right,trial];assert np.array_equal(np.isposinf(a),np.isposinf(b));ma_state=ma_prob=0.;exact_state_bits=True
  for start,end,kind in fieldschema:
   x=a[start:end];y=b[start:end];finite=np.isfinite(x);err=np.abs(x[finite].astype('f8')-y[finite].astype('f8'));atol,rtol=(.0002,.0002)if kind=='probability'else(.002,2e-5);assert np.all(err<=atol+rtol*np.abs(x[finite].astype('f8')))
   if kind=='probability':ma_prob=max(ma_prob,float(err.max(initial=0)))
   else:ma_state=max(ma_state,float(err.max(initial=0)));exact_state_bits=exact_state_bits and np.array_equal(ab[start:end],bb[start:end])
  pairs.append({'left':left,'right':right,'trial':trial,'status':'pass','max_abs_state_error':ma_state,'max_abs_probability_error':ma_prob,'all_state_payload_raw_bits_identical':bool(exact_state_bits),'exact_nonfinite_classes':True,'complete_fields_checked':len(fieldschema)})
summary={'benchmark':'prna','source_result':str(args.result),'source_result_sha256':hashlib.sha256(args.result.read_bytes()).hexdigest(),'configuration_id':result['configuration_id'],'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'NumPy_version':np.__version__,'status':'pass','models':models,'artifact_count':len(reports),'bytes_read_and_rehashed':sum(x['bytes']for x in reports),'all_four_models_present':len(models)==4,'actual_raw_field_representation_preserved':True,'no_canonicalized_NaN_agreement':True,'strict_budgets':plan['strict_field_budgets'],'central_envelope_alone_sufficient':False,'artifacts':reports,'complete_pair_trial_field_audits':pairs,'scientific_scope':'FullsourceCPUstates+actualfullGPUdualreferencearrays, independentn7Boltzmanncontrols, fullprobabilitynecessaryproperties; nofull292biologicalgroundtruth claim','GPU_executions':0};args.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in ['artifacts','complete_pair_trial_field_audits']},indent=2))
