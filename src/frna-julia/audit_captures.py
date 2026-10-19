"""CPU-only full serialized-contract audit. No GPU trials, no tolerance changes.
The complete-state crosscheck is source-derived; the separately enumerated five
MFE controls are independent mathematical certificates, not a full 77-base oracle.
"""
import argparse,array,hashlib,json,math,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=Path(__file__).resolve().parent
assert sys.byteorder=='little'
def words(name,kind):
 x=array.array(kind);x.frombytes((B/name).read_bytes());return x
params=words('params.bin','I');seq=words('sequences.bin','i');gold=words('serial_source_crosscheck.bin','i');guard=-1515870811
prefix=array.array('d',[2779096485]*32);prefix.extend(list(params));prefix.extend([2779096485]*32);k=1;cases=[]
for _ in range(seq[0]):
 n=seq[k];k+=1;s=seq[k:k+n];k+=n;cases.append(n);prefix.extend([guard]*32);prefix.extend(list(s));prefix.extend([guard]*32)
 for size,value in [(n*n,14000)]*4+[(n+1,-777777)]*2:
  prefix.extend([guard]*32);prefix.extend([value]*size);prefix.extend([guard]*32)
assert len(prefix)==128075 and k==len(seq)
expected=prefix[:];offset=0
for n in cases:
 for checkpoint in range(n):
  for size in [n*n]*4+[n+1]*2:
   expected.extend([guard]*32);expected.extend(list(gold[offset:offset+size]));expected.extend([guard]*32);offset+=size
assert offset==2205408 and len(expected)==2396459
raw=expected.tobytes();h=hashlib.sha256;result=json.loads(a.result.read_text());rows=[]
assert result['matched_performance_valid'] and all(result['variant_verified'].values()) and result['reference_agreement']
for model,m in result['models'].items():
 assert len(m['trials'])==5 and m['complete'] and m['timing_valid'] and m['provenance_valid']
 times=[t['time_seconds'] for t in m['trials']];assert math.isclose(sum(times[1:])/4,m['mean_seconds'],rel_tol=1e-15)
 for t in m['trials']:
  q=t['output_artifact'];f=Path(q['saved_path']);data=f.read_bytes();assert h(data).hexdigest()==q['sha256'];assert data==raw
  assert t['correct'] and t['checked_values']==2205408 and len(t['segment_seconds'])==164
  rows.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':h(data).hexdigest(),'bytes':len(data),'exact_input_prefix_values':128075,'exact_source_state_values':2205408,'all_state_and_input_guards_pass':True,'MFE_scaled_Int32':t['MFE_scaled_Int32']})
x={'all_pass':True,'result_path':str(a.result),'result_sha256':h(a.result.read_bytes()).hexdigest(),'five_trial_records':20,'artifact_values_each':len(expected),'actual_input_prefix_values':len(prefix),'full_artifact_hash_checks':rows,'independent_scientific_certificate':'small_structure_optima_review.json','source_crosscheck_not_independent_algorithm':True,'scientific_scope':'Five exhaustive controlled globaloptima; complete77base source-state agreement andterminal/prefixsuffix invariants, notbiologicalgroundtruth/globalindependent77base optimum','expected_artifact_sha256':h(raw).hexdigest()};tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(a.output);print(json.dumps({'all_pass':True,'artifacts':len(rows),'values_each':len(expected)}))
