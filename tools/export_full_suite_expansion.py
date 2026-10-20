#!/usr/bin/env python3
"""Join prospective full-suite records without altering the frozen 447 study."""
from pathlib import Path
import csv,datetime,hashlib,json,subprocess
from study_evidence import effective_config
from export_julia_300_registry import usage_windows
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'.porting-state/portable_julia';PAPER=ROOT/'2026-PASC-Hecbench-Julia-paper'
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def effort_manifest(b,model):
 aliases={'cuda_jl':['cuda_jl','cuda.jl','cuda'],'ka':['ka','KA'],'jacc':['jacc','JACC']}.get(model,[])
 paths=[ROOT/f'.porting-state/porting_logs/{b}-julia-{a}/manifest.json' for a in aliases]
 paths=[p for p in paths if p.is_file()]
 suffix={'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}.get(model)
 current=sha(ROOT/'src'/f'{b}-julia'/f'{b}-julia-{suffix}'/'main.jl') if suffix else None
 paths.sort(key=lambda p:(read(p,{}).get('source_sha256')==current,p.stat().st_mtime_ns),reverse=True)
 if paths and read(paths[0],{}).get('source_sha256')!=current:return {'effort_binding_status':'No manifest matches current main source; do not join stale candidate counts, costs or commits.'}
 return read(paths[0],{}) if paths else {}
def counter(d,*names):
 values={d[n] for n in names if type(d.get(n)) is int and d[n]>=0}
 return next(iter(values)) if len(values)==1 else None
def tex(s):return str(s).replace('_',r'\_').replace('&',r'\&')
def main():
 cohort=read(STATE/'batch_537.json');new=set(cohort['new_benchmarks']);latest={};partial={}
 for p in sorted((STATE/'runs').glob('*/result.json')):
  if p.parent.name.split('_',1)[-1] not in new:continue
  d=read(p);b=d.get('benchmark')
  if b not in new:continue
  c=effective_config(p,read(p.with_name('config.json'),{}))
  partial[b]=(d,c,p)
  if not c.get('exclude_from_aggregate',False) and c.get('configuration_role','primary') in ('primary','corrected_reference_secondary') and d.get('reference_agreement') and all(d.get('variant_verified',{}).get(m) for m in ('ka','jacc')):latest[b]=(d,c,p)
 rows=[]
 for b in sorted(new):
  record=latest.get(b) or partial.get(b);traits=read(STATE/'traits'/f'{b}.json',{});models={}
  row={'benchmark':b,'status':'unstarted_or_readiness_only','scientific_intent':traits.get('scientific_intent'),'shared_usage_windows':usage_windows(b,'shared'),'completion_scope_review':read(ROOT/'src'/(b+'-julia')/'completion_scope_review.json'),'models':models}
  if record:
   d,c,p=record;kind=c.get('numerical_agreement_kind','declared_numeric_artifact_contract');row.update(numerical_agreement_kind=kind,raw_output_identity_claim=c.get('raw_output_identity_claim',d.get('raw_output_identity_claim')),is_property=kind=='property_equivalence_not_identical_raw_values',is_execution_only=kind=='no_numerical_outputs_execution_only',is_numerical=kind not in ('property_equivalence_not_identical_raw_values','no_numerical_outputs_execution_only'),status='accepted_named_contract' if b in latest else 'in_progress_not_four_model_accepted',configuration_id=d['configuration_id'],result_path=str(p.relative_to(ROOT)),result_sha256=sha(p),protocol=d['protocol'],input_prefix_count=d.get('input_prefix_count'),reference_qualification='known_native_scientific_defect' if c.get('native_behavior_known_scientific_defect') else 'corrected_native_reference' if c.get('native_reference_repaired') else 'revised_contract' if c.get('verification_contract_revised') else 'multiple_device_named_contract' if str(c.get('gpu',6))!='6' else 'integration_or_scope_adapted' if c.get('exclude_untouched_native_aggregates') else 'ordinary_named_contract',timing_scope=c.get('timing_region') or (d.get('timing_region_metadata_adjudication')or{}).get('effective_timing_region') or next((t.get('timing_region') for v in d.get('models',{}).values() for t in v.get('trials',[]) if t.get('timing_region')),None),matched_performance_valid=d.get('matched_performance_valid',False),configuration_role=c.get('configuration_role','primary'),input_contract=c.get('input_contract'),native_reference_repaired=bool(c.get('native_reference_repaired')),native_behavior_known_scientific_defect=bool(c.get('native_behavior_known_scientific_defect')),verification_contract_revised=bool(c.get('verification_contract_revised')),exclude_untouched_native_aggregates=bool(c.get('exclude_untouched_native_aggregates')),reference_provenance=c.get('reference_provenance'),scientific_witness=read(STATE/'scientific_witnesses_300'/f'{b}.json'))
   for m in ('cuda','cuda_jl','ka','jacc'):
    v=d.get('models',{}).get(m,{});manifest=effort_manifest(b,m)
    models[m]={'status':v.get('status'),'complete':v.get('complete'),'mean_last_four_seconds':v.get('mean_seconds'),'five_trial_seconds':[t.get('time_seconds') for t in v.get('trials',[])],'all_five_checks_pass':len(v.get('trials',[]))==5 and all(t.get('correct') for t in v['trials']),'checked_values_per_trial':[t.get('checked_values') for t in v.get('trials',[])],'source_hash':v.get('source_hash'),'adapter_file_hashes':v.get('adapter_file_hashes'),'effort_binding_status':manifest.get('effort_binding_status','current_main_source_match' if manifest else 'missing'),'candidate_attempts':manifest.get('candidate_attempts'),'candidate_attempts_unit':manifest.get('candidate_attempts_unit',manifest.get('candidate_attempts_definition')),'authored_mathematical_algorithm_candidates':manifest.get('authored_mathematical_algorithm_candidates'),'source_transcription_generator_invocations':manifest.get('source_transcription_generator_invocations'),'device_kernel_definitions':manifest.get('device_kernel_definitions'),'benchmark_algorithm_device_kernel_definitions':manifest.get('benchmark_algorithm_device_kernel_definitions'),'timed_algorithm_kernel_launches_per_trial':manifest.get('timed_algorithm_kernel_launches_per_trial'),'untimed_parameter_observer_launches_per_trial':manifest.get('untimed_parameter_observer_launches_per_trial'),'compiler_checks':counter(manifest,'compiler_check_invocations','compiler_checks'),'actual_device_jit_process_checks':counter(manifest,'actual_device_jit_process_checks','authored_device_jit_processes','authored_device_JIT_check_programs'),'program_evaluation_unit':manifest.get('compiler_checks_unit'),'measurement_only_reruns':manifest.get('measurement_repair_evaluations'),'translation_phase_MPI_jobs':manifest.get('translation_phase_MPI_jobs'),'host_julia_process_checks':manifest.get('host_julia_process_checks'),'compiler_feedback_repairs':manifest.get('compiler_feedback_repairs'),'translation_tokens':manifest.get('translation_tokens'),'source_commit':manifest.get('source_commit'),'original_translation_commit':manifest.get('original_translation_commit'),'selected_execution_source':manifest.get('selected_execution_source'),'measurement_repair_commits':manifest.get('measurement_repair_commits'),'measurement_repair_results':manifest.get('measurement_repair_results'),'implementation_class':traits.get('implementation_traits',{}).get(m,{}).get('configuration_traits',{}).get(d['configuration_id'],{}).get('implementation_class') or traits.get('implementation_class',{}).get(m) if isinstance(traits.get('implementation_class',{}),dict) else traits.get('implementation_class'),'implementation_traits':{**traits.get('implementation_traits',{}).get(m,{}),**traits.get('implementation_traits',{}).get(m,{}).get('configuration_traits',{}).get(d['configuration_id'],{})},'configuration_traits_override_applied':bool(traits.get('implementation_traits',{}).get(m,{}).get('configuration_traits',{}).get(d['configuration_id']))}
  rows.append(row)
 accepted=[r for r in rows if r['status']=='accepted_named_contract'];cal=read(PAPER/'research/expansion_537/token_cost_calibration.json',{})
 doc={'snapshot_created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_native_names':537,'frozen_primary_population':447,'new_frontier_names':90,'new_four_model_accepted_names':len(accepted),'new_not_yet_four_model_accepted_names':90-len(accepted),'frozen_primary_analysis_changed':False,'published_v1_0_0_changed':False,'study_shared_usage_windows':usage_windows('study537','shared')+usage_windows('study90','shared'),'shared_window_scope':'Native readiness, common verification, annotation and paper/coordination windows remain separate and can overlap target windows; never add them as exclusive cost categories. Open meters live in active_work_537 and original BEGIN events.','frozen_primary_contracts':{'numerical':436,'property':2,'execution_only':5,'native_hardware_barrier':4},'expanded_accepted_numerical_contracts':436+sum(r.get('is_numerical',False) for r in accepted),'expanded_accepted_property_contracts':2+sum(r.get('is_property',False) for r in accepted),'expanded_execution_only_contracts':5+sum(r.get('is_execution_only',False) for r in accepted),'count_scope':'Qualified named contracts with numerical/property/execution outcomes distinct; includes explicit reference qualifications, not universal scientific validity or full-original-CLI validation of all537; scoped assessments may be complete while hardware cases remain unverified.','translation_order':'CUDAjl thenKA thenJACC within benchmark; shared helper and stream context inherited.','calibration':{'observed_windows':cal.get('observation_windows'),'eligible_windows':cal.get('eligible_windows'),'fitted_strata':sum(f.get('status')!='insufficient_calibration' for f in cal.get('fits',[])),'fits':[{k:f.get(k) for k in ('implementation_model','configured_llm','reasoning_effort_setting','contains_shared_harness_creation','distinct_benchmarks','status','point_estimate','leave_benchmark_out_median_absolute_percentage_error','empirical_total_token_error_band')} for f in cal.get('fits',[])],'historical_actuals_modified':False,'missing_historical_forecast_slots':len(cal.get('missing_historical_estimates',[])),'missing_new_frontier_forecast_slots':len(cal.get('missing_new_frontier_workflow_estimates',[])),'forecast_csv':'data/estimated_current_workflow_tokens.csv','scope':'Expected present-workflow forecasts separate from unknown historical actuals; insufficient per-stratum sample yields no estimate.'},'rows':rows}
 write(PAPER/'research/expansion_537/live_coverage.json',doc)
 compact={k:v for k,v in doc.items() if k!='rows'};compact['accepted_cases']=[{k:v for k,v in r.items() if k not in ('scientific_witness','reference_provenance')} for r in accepted]
 for r in compact['accepted_cases']:
  r['models']={m:{k:v for k,v in v.items() if k not in ('adapter_file_hashes','implementation_traits')} for m,v in r['models'].items()}
 write(PAPER/'data/prospective_extension.json',compact)
 out=PAPER/'sections/generated/full_suite_extension.tex';out.parent.mkdir(parents=True,exist_ok=True)
 lines=[r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{3pt}',r'\caption{Prospective named-region means (ms; last four of five checked trials). C: corrected reference; D: known defect; A: adapter; P: property contract; 2G: two devices. First eight alphabetical accepted names; all cases in compact data. Outside the frozen 447 analysis.}',r'\label{tab:prospective-extension}',r'\begin{tabular}{lrrrrl}',r'\toprule Case & CUDA & CUDA.jl & KA & JACC & Ref.\\',r'\midrule']
 for r in accepted[:8]:
  tag={'corrected_native_reference':'C','known_native_scientific_defect':'D','ordinary_named_contract':'ordinary','revised_contract':'R','integration_or_scope_adapted':'A','multiple_device_named_contract':'2G'}[r['reference_qualification']]
  if r.get('native_reference_repaired') and r.get('native_behavior_known_scientific_defect'):tag+=' C'
  if r.get('is_property'):tag+=' P'
  times=[format(1000*r['models'][m]['mean_last_four_seconds'],'.1f' if r['models'][m]['mean_last_four_seconds']>=1 else '.3f') for m in ('cuda','cuda_jl','ka','jacc')]
  lines.append(tex(r['benchmark'])+' & '+' & '.join(times)+' & '+tag+r' \\')
 outcome_parts=[str(sum(r.get(flag,False) for r in accepted))+' '+label for flag,label in [('is_numerical','numerical'),('is_property','property'),('is_execution_only','execution-only')] if any(r.get(flag,False) for r in accepted)]
 outcome_macro=r'\newcommand{\extensionOutcomeSummary}{'+' and '.join(outcome_parts)+'}'
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',outcome_macro,r'\newcommand{\extensionAccepted}{'+str(len(accepted))+'}',r'\newcommand{\extensionPending}{'+str(90-len(accepted))+'}'])
 out.write_text('\n'.join(lines)+'\n')

 fitted=[]
 for m,shared in (('cuda_jl',True),('ka',False),('jacc',False)):
  fit=next((f for f in cal.get('fits',[]) if f.get('implementation_model')==m and f.get('contains_shared_harness_creation') is shared and f.get('configured_llm')=='gpt-6.1-sol' and f.get('reasoning_effort_setting')=='medium' and f.get('status')=='exploratory_current_workflow_baseline'),None)
  if fit:fitted.append(fit)
 if len(fitted)==3:
  counts='/'.join(str(f['distinct_benchmarks']) for f in fitted)
  points='/'.join('{:.3f}'.format(f['point_estimate']['total_tokens']/1e6) for f in fitted)
  errors='/'.join('{:.0f}'.format(f['leave_benchmark_out_median_absolute_percentage_error']) for f in fitted)
  paragraph='The fitted CUDA.jl/KA/JACC strata contain '+counts+' distinct names, giving exploratory median-baseline input-plus-output forecasts of '+points+' million tokens with leave-one-benchmark-out median absolute percentage errors of '+errors+r'\%. These coarse forecasts apply to configured \texttt{gpt-6.1-sol}, medium effort, with shared-harness creation in the CUDA.jl stratum and ordered reuse in KA/JACC. They do not reconstruct historical actual consumption or establish target-cost superiority. Additional strata remain insufficient.'
 else:paragraph='Comparable strata remain below the minimum calibration floor; unknown historical actuals stay unknown and no fitted forecast is claimed.'
 (PAPER/'sections/generated/full_suite_calibration.tex').write_text(paragraph+'\n')

 details=PAPER/'companion/11_prospective_extension_details.tex'
 detail=[r'\section{Prospective Extension: Per-Case Effort}',r'\label{sec:extension-details}',r'These records concern the new frontier, outside the frozen 447 primary study. Tokens are context-inclusive input plus output (millions), with cached input and reasoning output retained as subsets in compact data. Program evaluations include reported reruns and are distinct from authored candidates. Two-rank MPI jobs and their rank processes are separately identified in compact data. Target order and shared-helper reuse prevent independent cost comparisons. Candidate units are retained per case where reported and are not assumed standardized when missing; program and GPU-JIT processes remain separate. Missing fields remain unknown.',r'\begin{table*}[t]',r'\centering\small',r'\caption{All accepted extension cases at this snapshot. Triplets are CUDA.jl/KA/JACC. Tokens are observed activity-window totals, not exclusive billing. Reference qualifications and complete five-trial arrays remain in the compact snapshot.}',r'\begin{tabular}{llllll}',r'\toprule Case & Candidates & Program evaluations & GPU-JIT processes & Tokens (M) & Reference \\',r'\midrule']
 for index,row in enumerate(accepted):
  if index and index%23==0:
   detail.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}',r'\begin{table*}[t]',r'\centering\small',r'\caption{Accepted extension cases, continued; definitions and qualifications as in the preceding table.}',r'\begin{tabular}{llllll}',r'\toprule Case & Candidates & Program evaluations & GPU-JIT processes & Tokens (M) & Reference \\',r'\midrule'])
  trip=lambda key: '/'.join('--' if row['models'][m].get(key) is None else str(row['models'][m][key]) for m in ('cuda_jl','ka','jacc'))
  costs='/'.join('--' if not row['models'][m].get('translation_tokens') or row['models'][m]['translation_tokens'].get('total_tokens') is None else '{:.3f}'.format(row['models'][m]['translation_tokens']['total_tokens']/1e6) for m in ('cuda_jl','ka','jacc'))
  tag={'ordinary_named_contract':'ordinary','corrected_native_reference':'corrected','known_native_scientific_defect':'known defect','revised_contract':'revised','integration_or_scope_adapted':'adapter','multiple_device_named_contract':'two-device'}[row['reference_qualification']]
  if row.get('native_reference_repaired') and row.get('native_behavior_known_scientific_defect'):tag='corrected; defect'
  detail.append(tex(row['benchmark'])+' & '+trip('candidate_attempts')+' & '+trip('compiler_checks')+' & '+trip('actual_device_jit_process_checks')+' & '+costs+' & '+tag+r' \\')
 detail.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
 details.parent.mkdir(parents=True,exist_ok=True);details.write_text('\n'.join(detail)+'\n')
 print(json.dumps({k:v for k,v in doc.items() if k in ['new_four_model_accepted_names','new_not_yet_four_model_accepted_names','expanded_accepted_numerical_contracts']}))
if __name__=='__main__':main()
