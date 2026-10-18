"""Complete independent CPU-octree witness + mass and logged structural checks."""
from pathlib import Path
import json,array,struct,math,hashlib,re,sys
P=Path(__file__).resolve().parent
CASES=[(256,1),(256,3),(1024,1),(1024,3),(2048,1)]
def floats(path,code):
 a=array.array(code);a.frombytes(Path(path).read_bytes());return a

def main():
 result=Path(sys.argv[1]).resolve();d=json.loads(result.read_text());expected=floats(P/'expected_f64.bin','d');prefix=array.array('f')
 for n,_ in CASES:prefix.extend(floats(P/f'fixture_{n}.bin','f'))
 records=[];maxerr=0.;percase=[];topologies={}
 for model,rec in d['models'].items():
  tr=rec.get('trials',[])
  if not tr:continue
  log=result.parent/f'{model}_run1.log'
  if not log.exists():log=Path(tr[0]['output_artifact']['saved_path']).parent/f'{model}_run1.log'
  lines=[line for line in log.read_text().splitlines()if line.startswith('STUDY_TREE_CHECK ')]
  assert len(lines)==25,(model,len(lines))
  assert all('topology=true'in line and 'guards=true'in line and 'allocations=13'in line for line in lines)
  topologies[model]={'raw_log':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'checked_final_trees':25,'checked_allocations_per_tree':13,'all_guards_exact_and_all_full_tree_invariants_pass':True,'raw_node_arrays_retained':False,'evidence_scope':'Complete final-tree DFS/count/start/sort/mass/finiteCOM selfchecks executed after every measured case; allocator-dependent node IDs are not public cross-model outputs. Exact guard bytes checked outside ROI, summarized in raw logs.'}
  for trial in tr:
   path=Path(trial['output_artifact']['saved_path']);a=floats(path,'f');assert len(a)==92160 and a[:46080].tobytes()==prefix.tobytes();out=a[46080:];pos=0;case_metrics=[]
   for n,steps in CASES:
    initial=floats(P/f'fixture_{n}.bin','f');state=out[pos:pos+10*n];ref=expected[pos:pos+10*n];pos+=10*n;errors=[abs(x-y)for x,y in zip(state,ref)];assert all(math.isfinite(x)and e<=1e-5+1e-4*abs(y)for x,y,e in zip(state,ref,errors));mx=max(errors);maxerr=max(maxerr,mx)
    assert all(struct.pack('<f',state[4*i+3])==struct.pack('<f',initial[4*i+3])for i in range(n));mass=math.fsum(state[4*i+3]for i in range(n));assert mass==1.
    def momentum(v):return[math.fsum(v[4*i+3]*(v[4*n+2*i+c]if c<2 else v[6*n+4*i+3])for i in range(n))for c in range(3)]
    before=momentum(initial);after=momentum(state);case_metrics.append({'n':n,'steps':steps,'full_public_state_values':10*n,'max_abs_error_vs_independent_CPU_octree':mx,'total_mass':mass,'all_body_mass_bits_preserved':True,'momentum_initial':before,'momentum_final':after,'momentum_drift_auxiliary_only':[y-x for x,y in zip(before,after)],'momentum_is_acceptance_criterion':False})
   records.append({'model':model,'trial':trial['trial'],'artifact':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_prefix_exact_bytes':True,'case_checks':case_metrics})
 witness={'benchmark':'bh','raw_result':str(result),'capture_count':len(records),'full_output_count':46080,'input_prefix_count':46080,'max_abs_error_vs_independent_CPU_octree':maxerr,'atol':1e-5,'rtol':1e-4,'oracle':{'source':str(P/'generate_oracle.py'),'source_sha256':hashlib.sha256((P/'generate_oracle.py').read_bytes()).hexdigest(),'expected_values':str(P/'expected_f64.bin'),'expected_sha256':hashlib.sha256((P/'expected_f64.bin').read_bytes()).hexdigest(),'method':'Independently constructed CPU spatial octree with child-order DFS and warp32 collective opening, higher precision COM/forces/integration and FP32-rounded public positions. Genuine tree approximation, not dense all-pairs surrogate.','scope_limit':'Same predeclared 5case finite-time operator/pipeline tolerance; not fullapplication, allCLI regimes or long-time accuracy/stability proof.'},'tree_guard_checks':topologies,'captures':records,'pass':len(records)==20}
 (P/'independent_witness.json').write_text(json.dumps(witness,indent=2)+'\n');print(json.dumps({k:v for k,v in witness.items()if k not in ['captures','tree_guard_checks']},indent=2))
if __name__=='__main__':main()
