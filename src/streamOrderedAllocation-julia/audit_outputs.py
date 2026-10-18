from pathlib import Path
from array import array
import json,hashlib,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];bp=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bp/'result.json').read_text());load=lambda q:array('d',Path(q).read_bytes());pre=load(p/'fixture.bin');gold=load(p/'oracle.bin');expected=pre+gold;rows=[];hashes=set();groups=[];go=0
for n in [1,255,257,4097]:
 for mode in range(3):groups.append(dict(n=n,mode=mode,start=go,outputs=100*n,inputs=2*n));go+=102*n
assert go==len(gold)
for m,z in d['models'].items():
 for t in z['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();v=array('d',raw);assert v==expected;assert raw[:8*len(pre)]==(p/'fixture.bin').read_bytes();sha=hashlib.sha256(raw).hexdigest();hashes.add(sha)
  for g in groups:
   start=len(pre)+g['start'];n=g['n'];out=v[start:start+100*n];a=v[start+100*n:start+101*n];b=v[start+101*n:start+102*n];numer=[(round(x*16)+round(y*16))/16 for x,y in zip(a,b)]
   for it in range(100):assert out[it*n:(it+1)*n]==array('d',numer)
  assert len(t['path_times_seconds'])==12 and abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
  rows.append(dict(model=m,trial=t['trial'],path=str(f),sha256=sha,count=len(v),all100outputs_eachpath_exact_integer_oracle=True,actual_final_GPU_input_echo_exact=True,all12times_sum_to_total=True))
assert len(rows)==20 and d['matched_performance_valid']
result=dict(run=run,all_pass=True,all20_full_artifacts=len(rows),words_per_artifact=len(expected),unique_artifact_hashes=sorted(hashes),outputs_each_trial=1383000,actual_final_inputecho_words_each_trial=27660,input_prefix_words=len(pre),groups=groups,rows=rows,original_GPU_guards_present=False,guard_protection_claim=False,max_abs_error=0,max_relative_error_nonzero_reference=0,zero_reference_absolute_error=0,arbitrary_float32_or_full33554432_CLI_not_certified=True)
(p/'output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','groups']}))
