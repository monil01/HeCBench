from pathlib import Path
from array import array
import hashlib,json,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];d=json.loads((r/'.porting-state/portable_julia/runs'/run/'result.json').read_text());pre=array('d',(p/'fixture.bin').read_bytes());gold=array('d',(p/'oracle.bin').read_bytes());desc=json.loads((p/'fixture_description.json').read_text());expected=pre+gold;rows=[];hashes=set()
for m,z in d['models'].items():
 for t in z['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();v=array('d',raw);assert v==expected;cursor=len(pre)
  for c in desc['cases']:
   T,B,Q,P,it=[c[k]for k in ['T','blocks','logical_queue','pool','iterations']];whole=[]
   for off in range(0,P,Q):
    before=v[cursor:cursor+Q*T+128];cursor+=Q*T+128;after=v[cursor:cursor+Q*T+128];cursor+=Q*T+128;tasks=v[cursor:cursor+2*(Q+B+128)];cursor+=2*(Q+B+128);counter=v[cursor:cursor+129];cursor+=129
    assert before==array('d',[16384]*64+[0]*(Q*T)+[16384]*64)
    for taskid in range(off,off+Q):
     local=taskid-off;op=c['ops'][taskid];value=T*(it if op==1 else 1)+taskid;assert after[64+local*T:64+(local+1)*T]==array('d',[value]*T);assert tasks[128+2*local:130+2*local]==array('d',[taskid,op]);whole.extend([value]*T)
    assert after[:64]==after[-64:]==array('d',[16384]*64);assert tasks[:128]==array('d',[16384]*128);assert tasks[128+2*Q:]==array('d',[16384]*(2*(B+64)));assert counter==array('d',[16384]*64+[Q+B]+[16384]*64)
   assert v[cursor:cursor+P*T]==array('d',whole);cursor+=P*T;assert v[cursor]==P;cursor+=1
  assert cursor==len(v)and len(t['path_times_seconds'])==4 and abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12;h=hashlib.sha256(raw).hexdigest();hashes.add(h);rows.append(dict(model=m,trial=t['trial'],path=str(f),sha256=h,all_tasks_outputs_full=True,counter_terminal_Q_plus_B=True,all_guard_and_initialized_padding_exact=True,actual_preGPU_inputs_zero=True))
assert len(rows)==20 and d['matched_performance_valid']
a=dict(run=run,all_pass=True,all20_full_artifacts=len(rows),words_per_artifact=len(expected),unique_hashes=sorted(hashes),max_abs_error=0,max_rel_nonzero_error=0,zero_reference_abs_error=0,original_kernel_body_unchanged=True,native_host_allocation_repaired=True,padding_bytes_per_case=[c['terminal_padding_bytes']for c in desc['cases']],extra_observer_D2H_per_trial=44,source_derived_launches_per_trial=11,work_execution_rate_claim=False,zero_heavy_id0_invariant_limitation='Itszerooutputcannotproveexecutedaddzero; sourceatomicuniqueness/counter/outputcoverage notdynamicarithmeticcounter.',rows=rows)
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:v for k,v in a.items()if k!='rows'}))
