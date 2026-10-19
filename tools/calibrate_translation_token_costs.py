#!/usr/bin/env python3
"""Export observed windows and separate, guarded expected-workflow token estimates."""
import argparse,csv,datetime,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COUNTERS=['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens']
MODELS=['cuda_jl','ka','jacc']

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def quantile(values,p):
    s=sorted(values)
    if not s:return None
    x=(len(s)-1)*p;i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)

def center(rows):
    d={k:statistics.median(r['usage'][k] for r in rows) for k in COUNTERS if k!='total_tokens'}
    d['total_tokens']=d['input_tokens']+d['output_tokens'];return d

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--minimum-benchmarks',type=int,default=8);a=ap.parse_args()
    if a.minimum_benchmarks<8:ap.error('At least eight distinct benchmark names are required; this is an exploratory safeguard, not sufficient evidence of generalization.')
    state=ROOT/'.porting-state/portable_julia';cohort=read(state/'batch_537.json');out=ROOT/'2026-PASC-Hecbench-Julia-paper/research/expansion_537';out.mkdir(parents=True,exist_ok=True)
    observations=[];groups=defaultdict(list)
    inventory=read(out/'readiness_inventory.json',{});complexity={r['benchmark']:r for r in inventory.get('rows',[])}
    for b in cohort['new_benchmarks']:
      for model in MODELS:
        f=state/'usage_boundaries_300'/b/(model+'_translation.jsonl');ev=[json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        adjud_path=state/'calibration_537'/b/(model+'.json');adj=read(adjud_path,{})
        for e in [e for e in ev if e.get('boundary') in ['end','abandon']]:
          u=e.get('delta_usage');llm=e.get('observed_model_metadata',{});llm=llm.get('model') if llm else None;llm=llm or adj.get('model_identifier')
          effort_setting=(e.get('observed_model_metadata') or {}).get('reasoning_effort') or adj.get('reasoning_effort')
          reasons=[]
          declared_b=adj.get('benchmark',adj.get('benchmark_id'))
          declared_m=adj.get('implementation_model',adj.get('model'))
          if declared_b is not None and declared_b!=b:reasons.append('adjudication_benchmark_identity_mismatch')
          if declared_m is not None and declared_m!=model:reasons.append('adjudication_implementation_identity_mismatch')
          if adj.get('end_event_id') is not None and adj['end_event_id']!=e['event_id']:reasons.append('adjudication_end_event_mismatch')
          if adj.get('begin_event_id') is not None and adj['begin_event_id']!=e.get('begin_event_id'):reasons.append('adjudication_begin_event_mismatch')
          if adj.get('observed_usage') is not None and any(adj['observed_usage'].get(k)!=((u or {}).get(k)) for k in COUNTERS):reasons.append('adjudication_actual_usage_mismatch')
          declared_path=next((adj[k] for k in ('usage_boundary_file','boundary_file','boundary_path') if adj.get(k)),None)
          if declared_path and (ROOT/declared_path).resolve()!=f.resolve():reasons.append('adjudication_window_path_mismatch')
          if e['boundary']!='end' or not u or any(u.get(k) is None or u[k]<0 for k in COUNTERS):reasons.append('incomplete_or_invalid_actual_window')
          if len([x for x in ev if x.get('boundary')=='end'])!=1 or any(x.get('boundary')=='abandon' for x in ev):reasons.append('multiple_or_partial_phase_segments_require_review')
          if adj.get('eligible_for_expected_workflow_cost_estimation') is not True:reasons.append('calibration_eligibility_not_adjudicated_true')
          if adj.get('cross_benchmark_work_inside_window') is not False:reasons.append('cross_benchmark_interleaving_unknown_or_present')
          if not isinstance(adj.get('contains_shared_harness_creation'),bool):reasons.append('shared_harness_stratum_unknown')
          if not llm:reasons.append('configured_model_identifier_unknown')
          if not effort_setting:reasons.append('configured_reasoning_effort_unknown')
          if u and all(isinstance(u.get(k),(int,float)) for k in COUNTERS):
            if u['total_tokens']!=u['input_tokens']+u['output_tokens'] or u['cached_input_tokens']>u['input_tokens'] or u['reasoning_output_tokens']>u['output_tokens']:reasons.append('counter_relationship_invalid')
          traits=read(state/'traits'/(b+'.json'),{})
          features={'native_source_lines':complexity.get(b,{}).get('code_lines'),'native_source_files':complexity.get(b,{}).get('code_files'),'scientific_family':(traits.get('scientific_intent') or {}).get('primary_family'),'candidate_attempts':adj.get('candidate_attempts'),'implementation_class':(traits.get('implementation_class') or {}).get(model) if isinstance(traits.get('implementation_class'),dict) else traits.get('implementation_class')}
          row={'features':features,'benchmark':b,'implementation_model':model,'configured_llm':llm,'reasoning_effort_setting':effort_setting,'agent':e.get('agent'),'event_id':e['event_id'],'source_identity_sha256':e.get('source_identity_sha256'),'usage':u,'marker_lag_seconds':e.get('marker_lag_seconds'),'observed_model_metadata':e.get('observed_model_metadata'),'shared_harness_creation':adj.get('contains_shared_harness_creation'),'context_reuse':adj.get('context_reuse'),'scope':e.get('scope'),'phase_adjudication':adj,'eligibility':not reasons,'exclusion_reasons':reasons,'window_path':str(f.relative_to(ROOT)),'window_file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'adjudication_sha256':hashlib.sha256(adjud_path.read_bytes()).hexdigest() if adjud_path.exists() else None}
          observations.append(row)
          if row['eligibility']:groups[(model,llm,effort_setting,row['shared_harness_creation'])].append(row)
    fits=[]
    for (model,llm,effort_setting,harness),rows in groups.items():
      distinct=len({r['benchmark'] for r in rows});fit={'implementation_model':model,'configured_llm':llm,'reasoning_effort_setting':effort_setting,'contains_shared_harness_creation':harness,'distinct_benchmarks':distinct,'agents':sorted({r['agent'] for r in rows}),'status':'insufficient_calibration','observed_benchmarks':sorted({r['benchmark'] for r in rows})}
      if distinct>=a.minimum_benchmarks:
        point=center(rows);errors=[];relative=[]
        for held in rows:
          train=[r for r in rows if r['benchmark']!=held['benchmark']];pred=center(train)['total_tokens'];actual=held['usage']['total_tokens'];errors.append(abs(pred-actual));relative.append(abs(pred-actual)/max(actual,1))
        q90=quantile(errors,.9);fit.update(status='exploratory_current_workflow_baseline',point_estimate=point,leave_benchmark_out_median_absolute_percentage_error=statistics.median(relative)*100,leave_benchmark_out_absolute_error_q90=q90,empirical_total_token_error_band=[max(0,point['total_tokens']-q90),point['total_tokens']+q90],interval_scope='Descriptive held-out-error band within observed workflow; not a calibrated confidence interval or independent generalization guarantee. Actor/context dependence remains.')
      fits.append(fit)
    # Counterfactual forecasts are separate from frozen measured/unknown fields.
    estimates=[]
    for b in cohort['baseline_447_names']:
      for model in MODELS:
        record=read(ROOT/'ontology/studies/julia_abstractions_447/implementations'/b/(model+'.json'),{});old=(record.get('effort') or {}).get('translation_tokens') or {}
        if old.get('historical_actual_total_tokens') is not None:continue
        choices=[f for f in fits if f['implementation_model']==model and f['status']=='exploratory_current_workflow_baseline']
        estimates.append({'benchmark':b,'implementation_model':model,'historical_actual_total_tokens':None,'historical_accounting_status':old.get('status'),'prior_saved_record_total_tokens':old.get('total_tokens'),'prior_saved_total_is_historical_actual':False,'historical_observed_activity_windows_exist':bool((record.get('effort') or {}).get('observed_harness_activity_windows')),'estimated_current_workflow_scenarios':choices,'estimation_status':'counterfactual_workflow_scenarios_available' if choices else 'insufficient_calibration','scope':'Expected declared translation/feedback-window tokens under each measured current workflow scenario, conditional on native readiness. Does not reconstruct actual historical token use; shared/native/verification/paper costs excluded. No observed ontology field overwritten.'})
    frontier_estimates=[]
    for b in cohort['new_benchmarks']:
      for model in MODELS:
        measured=any(r['benchmark']==b and r['implementation_model']==model and r.get('usage') and r['usage'].get('total_tokens') is not None for r in observations)
        if measured:continue
        choices=[f for f in fits if f['implementation_model']==model and f['status']=='exploratory_current_workflow_baseline']
        frontier_estimates.append({'benchmark':b,'implementation_model':model,'historical_actual_total_tokens':None,'historical_accounting_status':'new_frontier_no_closed_observed_translation_window','prior_saved_record_total_tokens':None,'estimated_current_workflow_scenarios':choices,'estimation_status':'conditional_future_workflow_scenarios_available' if choices else 'insufficient_calibration','scope':'Prospective conditional forecast for a missing target-window observation, not historical consumption or a successful-port guarantee. Same exploratory baseline; scientific-family/size/library/multiGPU applicability unvalidated.'})
    doc={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'minimum_distinct_benchmarks_per_model_llm_harness_stratum':a.minimum_benchmarks,'observation_windows':len(observations),'eligible_windows':sum(r['eligibility'] for r in observations),'method':'Componentwise median baseline within implementation/configured-LLM/shared-harness strata; total=input+output, cached/reasoning remain subsets; leave-one-benchmark-out descriptive error. New90-only training. No performance or historical model-effect claim.','limitations':['Activity windows include repeated/cached context and marker lag.','Inherited context, sequential target reuse, stopping rules and actor dependence limit generalization.','No token totals inferred from saved source length.','Counterfactual workflow estimates remain separate from unknown historical consumption.','Eight samples is an exploratory floor, not proof of predictive accuracy.','This pilot baseline does not condition on scientific family, implementation class or source complexity; those recorded covariates require a larger validated model.'],'observations':observations,'fits':fits,'missing_historical_estimates':estimates,'missing_new_frontier_workflow_estimates':frontier_estimates}
    (out/'token_cost_calibration.json').write_text(json.dumps(doc,indent=2)+'\n')
    with (out/'token_observations.csv').open('w',newline='') as f:
      fields=['benchmark','implementation_model','configured_llm','reasoning_effort_setting','agent','event_id','eligibility','marker_lag_seconds']+COUNTERS;w=csv.DictWriter(f,fields);w.writeheader()
      for r in observations:w.writerow({k:r.get(k) for k in fields if k not in COUNTERS}|{k:(r['usage'] or {}).get(k) for k in COUNTERS})
    publication=ROOT/'2026-PASC-Hecbench-Julia-paper/data';publication.mkdir(parents=True,exist_ok=True)
    with (publication/'estimated_current_workflow_tokens.csv').open('w',newline='') as f:
      fields=['benchmark','implementation_model','historical_actual_total_tokens','prior_record_status','prior_saved_proxy_total_tokens','configured_llm','reasoning_effort_setting','contains_shared_harness_creation','calibration_distinct_benchmarks','estimated_current_workflow_input_tokens','estimated_current_workflow_output_tokens','estimated_current_workflow_total_tokens','empirical_error_band_lower','empirical_error_band_upper','leave_benchmark_out_median_absolute_percentage_error','estimate_kind','forecast_population','applicability'];w=csv.DictWriter(f,fields);w.writeheader()
      for record in estimates+frontier_estimates:
        for fit in record['estimated_current_workflow_scenarios']:
          point=fit['point_estimate'];band=fit['empirical_total_token_error_band']
          w.writerow({'benchmark':record['benchmark'],'implementation_model':record['implementation_model'],'historical_actual_total_tokens':None,'prior_record_status':record['historical_accounting_status'],'prior_saved_proxy_total_tokens':record['prior_saved_record_total_tokens'],'configured_llm':fit['configured_llm'],'reasoning_effort_setting':fit['reasoning_effort_setting'],'contains_shared_harness_creation':fit['contains_shared_harness_creation'],'calibration_distinct_benchmarks':fit['distinct_benchmarks'],'estimated_current_workflow_input_tokens':point['input_tokens'],'estimated_current_workflow_output_tokens':point['output_tokens'],'estimated_current_workflow_total_tokens':point['total_tokens'],'empirical_error_band_lower':band[0],'empirical_error_band_upper':band[1],'leave_benchmark_out_median_absolute_percentage_error':fit['leave_benchmark_out_median_absolute_percentage_error'],'estimate_kind':'current_workflow_counterfactual_not_historical_actual','forecast_population':'frozen447_missing_historical_actual' if record['benchmark'] in cohort['baseline_447_names'] else 'new90_missing_closed_window','applicability':'Unvalidated across scientific family/complexity/library/multiGPU class; conditional on native readiness and ordered target reuse; not measured benchmark consumption.'})
    print(json.dumps({'observed_windows':len(observations),'eligible_windows':sum(r['eligibility'] for r in observations),'fitted_strata':sum(f['status']=='exploratory_current_workflow_baseline' for f in fits),'historical_actuals_modified':False}))

if __name__=='__main__':main()
