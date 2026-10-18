"""Read-only full record/source/log audit; no GPU executions or numerical claims."""
import pathlib,json,hashlib,math,statistics
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent;S=R/'.porting-state/portable_julia'
F=S/'runs/20261018T140957632098Z_assert/result.json';r=json.load(open(F));supp=json.load(open(S/'diagnostics/assert/supplemental/observations.json'))
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def ev(f):return {'path':str(pathlib.Path(f).relative_to(R)),'sha256':sha(f)}
assert r['matched_performance_valid'] and all(r['variant_verified'].values()) and r['reference_agreement']
rows=[];bindings={}
for m,v in r['models'].items():
 assert v['provenance_valid'] and v['timing_valid'] and len(v['trials'])==5
 for f,h in v['adapter_file_hashes'].items():assert sha(f)==h;bindings[f]=h
 for t in v['trials']:
  assert t['observed_device_assertion_status']==t['expected_device_assertion_status']==710
  for f in ['assertion_failure_context_isolated','execution_checked','correct','device_completion_checked','jit_excluded','device_assertions_present','successful_assertion_kernels_checked','expected_device_assertion_failure_checked']:assert t[f] is True
  assert t['checked_values']==0 and t['max_abs_error'] is None and t['max_rel_error'] is None
  assert 'output_artifact' not in t and not t.get('numerical_outputs') and not t.get('values')
  assert math.isclose(sum(t['component_time_seconds']),t['time_seconds'],rel_tol=1e-14)
  assert t['negative_diagnostic']['observed_status']==710 and t['negative_diagnostic']['process_returncode']==0
  rows.append({'model':m,'trial':t['trial'],'time_seconds':t['time_seconds'],'component_time_seconds':t['component_time_seconds'],'actual_negative_status':710,'primary_raw_child_text_available':False,'original_child_log_sha256':t['negative_diagnostic']['sha256'],'original_deleted_temporary_log':t['negative_diagnostic']['log'],'numerical_agreement_claim':False})
 assert math.isclose(statistics.mean([t['time_seconds']for t in v['trials'][1:]]),v['mean_seconds'],rel_tol=1e-14)
assert len(supp)==20
for x in supp:
 assert x['actual_statuses']==[710] and x['returncode']==0
 for f in ['stdout','stderr']:assert sha(R/x[f]['path'])==x[f]['sha256']
 x['retained_stdout']=(R/x['stdout']['path']).read_text();x['retained_stderr']=(R/x['stderr']['path']).read_text()
D=S/'diagnostics/assert';codegen=[]
for name in ['original_sm70.ptx','original_sm70_sass.txt','original_run.log','original_build.log','original_ptx_build.log','supplemental_build.log']:
 f=D/name;codegen.append(ev(f))
ptx=(D/'original_sm70.ptx').read_text();sass=(D/'original_sm70_sass.txt').read_text()
assert '__assertfail' in ptx and 'CALL.ABS.NOINC' in sass and 'BRA' in sass
out={'result':ev(F),'validation_kind':'no_numerical_outputs_execution_only','numerical_agreement_claim':False,'numerical_accuracy':'not_applicable_source_has_no_stores','roofline_applicable':False,'records':rows,'means_seconds':{m:v['mean_seconds']for m,v in r['models'].items()},'exact_measured_source_bindings':bindings,'source_contract':ev(P/'source_contract.json'),'supplemental_negative_diagnostics':supp,'supplemental_meaning':'Twenty newly observed isolated contexts corroborate actual710 with complete logs. They are not recovery of deleted original child text, not same executions and do not replace any primary timings. Extra process/JIT/build effort separate.','original_log_retention_deviation':'Original child status/exit/isolation/hash attestations retained; full raw text was in removed temporary directories and unavailable. Root-approved fresh corroboration preserves immutable original status/timing series.','native_codegen_evidence':codegen,'native_codegen_observation':'Original CUDA12.0 sm70 PTX retains __assertfail and loops; linked SASS has predicate branches and CALL instructions in both positive kernels. No claim about dynamic assertion count or Julia emitted instruction count.','target_codegen_scope':'Three Julia source bodies/mappings reviewed; actual device assertion ABI proved by status710/logs, per-model linked SASS not captured; no optimizer-elimination claim.'}
(P/'execution_timing_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('FULL20_PRIMARY_STATUS_TIMING_AND20_NEW_RAW_DIAGNOSTICS_AUDITED',len(bindings))
