"""Check every captured value and derive accuracy curves without rerunning GPUs."""
from pathlib import Path
import math,struct,json,hashlib
R=Path(__file__).resolve().parents[2];B=Path(__file__).parent;S=R/'.porting-state/portable_julia/runs';N=8269;PS=list(range(1,17))+[18,20,23]
H=struct.unpack('<8269f',(B/'study_input_f32.bin').read_bytes());REF=[math.log2(x) for x in H]

def ulp32(x):
 u=struct.unpack('<I',struct.pack('<f',x))[0];return struct.unpack('<f',struct.pack('<I',u+1))[0]-struct.unpack('<f',struct.pack('<I',u))[0]

def metrics(data):
 values=struct.unpack('<165401f',data);assert data[:N*4]==(B/'study_input_f32.bin').read_bytes();assert list(values[N:N+19])==PS
 out=values[N+19:];assert out[0]==-12345.5 and out[-1]==-12345.5;rows=[]
 for row,p in enumerate(PS):
  errors=[abs(x-y) for x,y in zip(out[1+row*N:1+(row+1)*N],REF)];budgets=[2.**(-p)+16*2.**(-24)+2*ulp32(x) for x in REF]
  rows.append({'requested_bits':p,'max_absolute_log2_error':max(errors),'rmse':math.sqrt(sum(x*x for x in errors)/N),'pointwise_budget_failures':sum(e>b for e,b in zip(errors,budgets)),'nominal_quantization_bound':2.**(-p),'maximum_applied_budget':max(budgets)})
 return rows

def audit():
 original=json.loads((S/'20261018T035221482758Z_log2/result.json').read_text());finalpath=S/'20261018T035832120775Z_log2/result.json';final=json.loads(finalpath.read_text());assert final['matched_performance_valid'];checks=[];original_curve=None;corrected_curve=None
 for t in original['models']['cuda']['rejected_trials']:
  a=t['output_artifact'];raw=Path(a['saved_path']).read_bytes();curve=metrics(raw);assert sum(x['pointwise_budget_failures'] for x in curve)==66574;assert hashlib.sha256(raw).hexdigest()==a['sha256'];assert original_curve in [None,curve];original_curve=curve
 for m,r in final['models'].items():
  for t in r['trials']:
   a=t['output_artifact'];raw=Path(a['saved_path']).read_bytes();curve=metrics(raw);assert sum(x['pointwise_budget_failures'] for x in curve)==0;assert hashlib.sha256(raw).hexdigest()==a['sha256'];assert corrected_curve in [None,curve];corrected_curve=curve;checks.append({'model':m,'trial':t['trial'],'pass':True,'artifact_sha256':a['sha256'],'values':165401,'exact_input_prefix_bytes':N*4,'host_precision_argument_metadata_values':19})
  assert len(r['trials'])==5 and math.isclose(sum(t['time_seconds'] for t in r['trials'][1:])/4,r['mean_seconds'],rel_tol=1e-11)
 # Clear independent witness, selected from known input1.5, not a fitted tolerance.
 i=H.index(1.5);assert abs(REF[i]-math.log2(1.5))<1e-15
 d={'configuration_id':final['configuration_id'],'accepted_result':str(finalpath.relative_to(R)),'accepted_result_sha256':hashlib.sha256(finalpath.read_bytes()).hexdigest(),'accepted_checks':checks,'accepted_artifact_unique_hashes':len({x['artifact_sha256'] for x in checks}),'original_rejected_captures':5,'original_pointwise_budget_failures':66574,'original_curve':original_curve,'corrected_curve':corrected_curve,'independent_oracle':'Python math.log2 on exactFloat32inputs storedFloat64; libmnotMPFR/formalintervalcertification. Original/correctedallGPUvalues comparedpointwise, notonlyRMSE.','scope':'8269finitepositiveinputswithinoriginalrange,19requestedprecisionlevels; noformalexhaustiveall-inputcertification. Algorithmrequestedbits differfromFloat32storage.','mean_seconds':{m:r['mean_seconds'] for m,r in final['models'].items()},'issues':[]}
 (B/'output_audit.json').write_text(json.dumps(d,indent=2)+'\n');print({'accepted_captures':20,'original_rejected_captures':5,'unique_accepted_hashes':d['accepted_artifact_unique_hashes'],'issues':0,'means':d['mean_seconds']})
if __name__=='__main__':audit()
