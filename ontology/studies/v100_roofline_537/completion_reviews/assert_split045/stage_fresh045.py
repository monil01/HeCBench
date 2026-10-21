"""CPU-only path relocation of frozen ASSERT037 phases; no command execution."""
import copy,difflib,hashlib,json
from pathlib import Path
OLD=Path('/tmp/hecbench-roofline-v100-20261020/batch037-assert-isolated-phases');NEW=Path('/tmp/hecbench-roofline-v100-20261020/batch045-assert-isolated-phases');OWN=Path(__file__).resolve().parent
OLDREVIEW=Path('/tmp/roofline-completion-20261021/assert_split037/isolated_assertion_review.json');REVIEW=OWN/'isolated_assertion_review.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def relocate(v):
 if isinstance(v,str):return v.replace(str(OLD),str(NEW)).replace(str(OLDREVIEW),str(REVIEW))
 if isinstance(v,list):return[relocate(x)for x in v]
 if isinstance(v,dict):return{relocate(k):relocate(x)for k,x in v.items()}
 return v
if NEW.exists():raise RuntimeError('fresh batch required')
oldreview=json.loads(OLDREVIEW.read_text());review=relocate(copy.deepcopy(oldreview));proofs=[];pending=[]
for model in ['cuda_jl','ka','jacc']:
 prior=OLD/'assert'/model;slot=NEW/'assert'/model;slot.mkdir(parents=True);scratch=slot/'scratch'
 for sub in ['', 'profile_outputs','ncu_tmp']:(scratch/sub).mkdir(exist_ok=True)
 oldplan=json.loads((prior/'plan.json').read_text());oldspec=json.loads((prior/'phase_spec.json').read_text())
 for p,h in oldplan['execution_bindings'].items():
  if sha(p)!=h:raise RuntimeError('037 original binding drift '+p)
 if sha(OLDREVIEW)!=oldplan['isolated_assertion_review']['sha256']:raise RuntimeError('037 review mismatch')
 spec=relocate(copy.deepcopy(oldspec));dump(slot/'phase_spec.json',spec);d=relocate(copy.deepcopy(oldplan));record=review['models']['assert/'+model]
 record['phase_spec']['sha256']=sha(slot/'phase_spec.json');record['execution_bindings'][str(slot/'phase_spec.json')]=sha(slot/'phase_spec.json')
 d['execution_bindings'][str(slot/'phase_spec.json')]=sha(slot/'phase_spec.json')
 # Same exact original helper, project/ELF, command, config/result and source hashes.
 if oldspec['execution_bindings']!=spec['execution_bindings']:raise RuntimeError('unexpected helper/source change')
 if oldspec['positive_command']!=spec['positive_command']:raise RuntimeError('positive argv changed')
 proof={'model':model,'old_plan':{'path':str(prior/'plan.json'),'sha256':sha(prior/'plan.json')},'old_phase_spec_sha256':sha(prior/'phase_spec.json'),'old_review_sha256':sha(OLDREVIEW),'new_phase_spec_sha256':sha(slot/'phase_spec.json'),'source_bindings_identical':True,'positive_command_identical':True,'source_config_result_helper_project_ELF_unchanged':True,'authorization_marker_unchanged':True,'changes':'Fresh scratch/receipt/spec/review paths only; no assertion/body/check/geometry/timing or method changes. Parent normalizes collector source-root absolute separately.','old_failure_preserved':True,'old_failure_class':'collector_relative_guard_tool_path_before_negative_context_launch; not compiler/framework/numerical failure','GPU_build_tests':0}
 for name in ['profile_provenance.json','pre_profile_checks.log','isolated_assertion_negative_raw.json']:
  p=prior/name
  if p.is_file():proof.setdefault('prior_actual_receipts',[]).append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 proofs.append(proof);pending.append((oldplan,d,slot))
dump(REVIEW,review)
for old,d,slot in pending:
 for key in ['review','isolated_assertion_review']:
  d[key]['path']=str(REVIEW);d[key]['sha256']=sha(REVIEW)
 d['execution_bindings'][str(REVIEW)]=sha(REVIEW)
 extra=d.get('review',{}).get('entry',{}).get('additional_current_readonly_input_bindings',{})
 if isinstance(extra,dict):
  if str(slot/'phase_spec.json')in extra:extra[str(slot/'phase_spec.json')]=sha(slot/'phase_spec.json')
  if str(REVIEW)in extra:extra[str(REVIEW)]=sha(REVIEW)
 d['fresh_isolated_assertion_relocation']={'prior_plan_sha256':sha(OLD/'assert'/d['model']/'plan.json'),'stage_helper':str(Path(__file__).resolve()),'stage_helper_sha256':sha(__file__),'old_failures_unchanged':True,'collector_absolute_source_root_required':True,'GPU_build_tests':0}
 dump(slot/'plan.json',d)
 dump(slot/'plan_diff.json',{'old_plan_sha256':sha(OLD/'assert'/d['model']/'plan.json'),'new_plan_sha256':sha(slot/'plan.json'),'diff':''.join(difflib.unified_diff(json.dumps(old,indent=2).splitlines(True),json.dumps(d,indent=2).splitlines(True),fromfile=str(OLD/'assert'/d['model']/'plan.json'),tofile=str(slot/'plan.json'))),'source_files_changed':[]})
 for p,h in d['execution_bindings'].items():
  if sha(p)!=h:raise RuntimeError('fresh binding mismatch '+p)
 for p,h in review['models']['assert/'+d['model']]['execution_bindings'].items():
  if d['execution_bindings'].get(p)!=h:raise RuntimeError('phase review not bound by plan')
row={'schema_version':1,'review':{'path':str(REVIEW),'sha256':sha(REVIEW)},'stage_helper':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'original_phase_helper':{'path':'/tmp/roofline-completion-20261021/assert_split037/assert_split_phases.py','sha256':sha('/tmp/roofline-completion-20261021/assert_split037/assert_split_phases.py')},'plans':[{'model':d['model'],'path':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'phase_spec_sha256':sha(slot/'phase_spec.json')}for _,d,slot in pending],'source_scope_proofs':proofs,'GPU_build_tests_executed':False,'old037_unmodified':True,'root_launch_requirement':'Absolute --source-root /home/imo/HeCBench_julia and --isolated-assertion-reviewed; root-reviewed collector path normalization must be adopted separately. All5negative prechecks/directpositive5PERF/postjoin mandatory.'}
dump(OWN/'delivery.json',row);print(json.dumps({'review_sha256':sha(REVIEW),'plans':row['plans']}))
