#!/usr/bin/env python3
"""Read-only all537x4 V100 selected-stage assessment; writes fresh /tmp only.
Successful selected kernel observations are summarized by stage/precision/tier;
no equal-kernel averages, missing-as-zero, complete host-ROI or causal inference.
"""
import argparse,collections,contextlib,csv,fcntl,gzip,hashlib,json,math,pathlib
P=pathlib.Path
MODELS=('cuda','cuda_jl','ka','jacc');REGIMES={'memory_side','near_ridge','compute_side'}
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def hbytes(b):return hashlib.sha256(b).hexdigest()
def number(x):return float(x)if isinstance(x,(int,float))and math.isfinite(x)else None
def counts(x):return dict(sorted(collections.Counter(x).items()))
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def extrema(v):
 v=[number(x)for x in v if number(x)is not None]
 return {'known_count':len(v),'min':min(v)if v else None,'max':max(v)if v else None}
def phase(ctx):
 if ctx.get('semantic_family_in_primary_ROI')is False:return 'explicit_source_exclusion'
 if ctx.get('semantic_family_in_primary_ROI')is not True:return 'source_membership_unassigned_or_candidate'
 if ctx.get('sample_in_primary_timed_invocation')is True and ctx.get('retained_trial'):return 'source_counted_retained_invocation_not_whole_elapsed_ROI'
 if ctx.get('first_invocation_may_be_warmup')is True:return 'first_invocation_or_warmup_qualified'
 return 'source_family_member_exact_invocation_unproved'
def projections(k,slotapp):
 out=[];scalar=k.get('scalar_precision_projections_by_memory_tier')or{}
 if not scalar:scalar={'dram':k.get('scalar_precision_projections',{})}
 for tier,precisions in scalar.items():
  for precision,p in precisions.items():
   work=p.get('executed_scalar_flops',p.get('executed_flops'));out.append((precision,tier,work,p))
 for tier,p in(k.get('tensor_precision_projections')or{}).items():out.append(('tensor',tier,p.get('executed_flops'),p))
 rows=[]
 for precision,tier,work,p in out:
  regime=p.get('roofline_regime');app=slotapp
  if tier=='dram'and app.startswith('not_applicable'):admit=False;reason='primary_memory_domain_not_DRAM_applicable'
  elif number(work)is None:admit=False;reason='executed_work_unknown_or_uncalibrated'
  elif work==0:admit=False;reason='no_counted_work_for_this_precision'
  elif regime not in REGIMES:admit=False;reason='undefined_geometric_envelope_or_missing_counter'
  else:admit=True;reason='qualified_projection_only'
  rows.append({'precision':precision,'memory_tier':tier,'executed_work':work,'geometric_regime':regime,'positioning_admitted':admit,'applicability_reason':reason,'memory_domain_qualification':app if tier=='dram'else'L2_transaction_tier_not_a_full_hierarchy_model','intensity':p.get('arithmetic_intensity_flop_per_byte',p.get('dram_arithmetic_intensity_flop_per_byte')),'intensity_over_ridge':p.get('intensity_over_ridge'),'achieved_kernel_replay_flops_per_second':p.get('achieved_flops_per_second'),'projection_only':p.get('projection_only',True),'causal_bottleneck':'unknown'})
 return rows
PROTOCOL='''# All537 benchmark/model selected-stage assessment

This is an exhaustive **assessment registry**, not a claim that all537 have
complete GPU ROI profiles or that any geometric label establishes a bottleneck.
The denominator is 537 names x four models = 2148 slots, preserving 1611 Julia
implementation slots and the separate native baseline. Hardware prerequisites,
failed processes, plan/replay blocks, pending collection, missing numerical or
source/observation bindings remain explicit. Missing measurements are null.

## Admission and provenance

Read current ledgers under the coordinator shared metadata lock. Bind protocol,
source census, scientific-domain record, selected ledger, original accepted
result/config, frozen profile manifest and selected annotation hashes. Verify
compressed AND uncompressed manifest hashes, plan/provenance joins and successful
selected process; exclude failed-process counter fragments from positioning.
Selected profile-ID directory, manifest path and observation-link directory must
agree. A stale top-level ledger provenance SHA is recorded explicitly, never
silently used: the actual selected manifest/provenance/counter/config/source join
is authoritative only when all those independent links and checks agree.
Rehash current declared source bindings (once per distinct file); expose drift
or unavailable source separately from measured old evidence. A before/after
source-bound capture is not invalidated merely because current tool/source code
has since changed, but current reproduction and paired interpretation must then
be qualified. Current original primary artifact bytes are not re-audited here.

## What is measured

Keep every selected valid kernel observation, including source-unassigned,
first-invocation/warmup and explicit setup/validation exclusions. Group exact
semantic-stage labels with their proof level, scientific family and observed
precision class. Candidate identifiers are not equivalent to executed semantic
stages. Do not infer arithmetic precision from application/storage labels.
Scalar FP16/FP32/FP64 projections and tensor projections remain separate, as do
DRAM and L2. Uncalibrated tensor/half work, no counted FP, missing counters,
zero-traffic intensity and host-mapped/PCIe memory are distinct applicability
outcomes. Retain diagnostics even when a DRAM prediction is inapplicable.

Report distributions of **kernel observations/projections**, never rename those
counts as benchmark counts. Ranges and category counts are summaries, not mean
kernel intensity/throughput/occupancy. Repeated launches, specializations and
geometry count multiplicities are preserved. No per-benchmark regime is inferred
from a majority vote of sampled kernels. Integer, address, atomic, SFU, host,
communication and allocation costs are not fully represented by scalar FLOPs.
All cold-cache kernel replay data are distinct from ordinary five-run timing.

## Timing, comparisons and aggregation

The accepted primary mean is reported separately, retaining configuration,
outcome/reference qualification and normalization scope. It is never a rate
denominator for fresh hardware counters. An ordinary family match or kernel-name
match cannot establish cross-model stage/case equivalence, actual-input equality,
precision equivalence, phase completeness, or temporal overlap.

Native/target positioning comparisons here require both captures to pass an
external, hash-bound **strict source-counted complete-family audit**, with the
same configuration, source ROI sequence, launch count, precision, counter/cache
policy and scientific numerical contract. Such pairs compare only integrated
work/traffic coordinates for the same named device family; counter-derived time
ratios remain cold-replay device-service ratios, not host performance speedups.
A primary timing ratio is not joined to stage-only samples. All other pairs are
explicitly insufficient, not zero. No cross-model source callsite identifier is
matched merely by similar spelling, order or callback name.

Additional full-family aggregation requires an explicit source launch ledger,
retained invocation selector, exact emitted match count, all required stage
coverage and numerical/self-check completion; one sample cannot be multiplied
by source loop count. Only declared triad/BabelStream-dot/Jacobi receipts are
currently accepted by the separate strict family exporter. Other finished
processes are candidates needing new source/selector proof, not ready aggregates.
Even a complete Jacobi stencil family omits timed reset/copy/host work. No complete
elapsed-host ROI or causal bottleneck is assigned by this assessment.

## Compact chart recommendations

1. Four horizontal status bars with the same 537-name denominator, explicitly
   separating valid selected observations from failed, blocked and pending.
2. Domain x model small multiples showing *names with any qualified projection*
   plus unknown/missing counts. This avoids weighting a name by launch count.
3. Kernel-level log I/ridge distributions faceted by precision and memory tier;
   separate source-proven retained, first/warmup and unassigned samples, label
   observation counts and expose host-tier exclusions. Do not pool scalar/tensor.
4. Only for admitted source-counted native/target families, paired integrated
   intensity points, with per-family output/tolerance/scope labels. Do not fit
   performance correlations or draw causal selection recommendations from the
   current tiny compatible subset. Preserve old controlled Triad plots separately.
'''
def compact_descriptors(payload):
 rows=[]
 for s in payload['model_assessments']:
  stages=s['stage_assessments'];qualified=sum(p['qualified_positioning_count']for st in stages for p in st['precision_tier_positioning'])
  if s['scientific_family_aggregate']:confidence='source_counted_complete_scientific_family_not_elapsed_ROI'
  elif s['assessment_category']=='source_reviewed_runtime_without_profiled_device_kernels':confidence='source_reviewed_runtime_no_kernel_FP_roofline_applicable'
  elif s['selected_valid_kernel_observation_count']is None:confidence='unmeasured_or_blocked_no_kernel_position_claim'
  elif s['assessment_category']=='valid_selected_observations_no_counted_FP':confidence='no_conventional_FP_counted_not_a_scalar_FP_bottleneck_claim'
  elif not any(st['source_stage_is_proved']for st in stages):confidence='measured_kernel_geometry_source_stage_unresolved'
  else:confidence='measured_source_family_geometry_invocation_qualified'
  rows.append({'benchmark':s['benchmark'],'model':s['model'],'scientific_family':s['scientific_family'],'configuration_id':s['configuration_id'],'profiling_status':s['profiling_status'],'assessment_category':s['assessment_category'],'positioning_confidence_descriptor':confidence,'valid_selected_kernel_observations':s['selected_valid_kernel_observation_count'],'semantic_stage_groups':len(stages),'qualified_precision_tier_projections':qualified if s['selected_valid_kernel_observation_count']is not None else None,'phase_qualification_distribution':s.get('phase_qualification_distribution'),'precision_class_distribution':s.get('precision_class_distribution'),'primary_unprofiled_mean_seconds':s['current_primary_mean_seconds'],'scientific_family_aggregate_available':s['scientific_family_aggregate']is not None,'capture_qualifications':s['observation_qualifications'],'complete_elapsed_GPU_or_host_ROI':False,'causal_bottleneck':'unknown'})
 return {'schema_version':1,'denominators':payload['denominators'],'registry_assessment_complete':True,'GPU_ROI_campaign_complete':False,'causal_bottleneck_campaign_complete':False,'per_model':payload['per_model'],'records':rows,'descriptor_counts':counts(r['positioning_confidence_descriptor']for r in rows),'confidence_labels_are_evidence_scope_not_probabilities':True,'projection_counts_are_not_benchmark_counts':True}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--retained-family-audit',type=P);ap.add_argument('--offline-snapshot',action='store_true');a=ap.parse_args();root=a.source_root.resolve();out=a.output_root.absolute()
 if not out.resolve().is_relative_to(P('/tmp'))or out.resolve()==P('/tmp')or out.exists()or out.is_symlink():ap.error('Fresh directory strictly inside /tmp required')
 inputs={};sourcechecks={};readcache={};issues=[];slots=[];kernelrows=[];stagecsv=[];pairstats=collections.Counter();pairs=[]
 def path(p):p=P(p);return p if p.is_absolute()else root/p
 def bind(p,expected=None,role='input'):
  p=path(p);key=str(p)
  if key not in inputs:inputs[key]={'sha256':sha(p),'bytes':p.stat().st_size,'roles':[]}
  if role not in inputs[key]['roles']:inputs[key]['roles'].append(role)
  if expected and inputs[key]['sha256']!=expected:raise ValueError('Hash mismatch '+key)
  return p
 def read(p,expected=None,role='input'):
  p=bind(p,expected,role);key=str(p)
  if key not in readcache:readcache[key]=p.read_bytes()
  return readcache[key]
 def load(p,expected=None,role='input'):return json.loads(read(p,expected,role))
 def sourcecheck(p,expected):
  key=(str(path(p)),expected)
  if key not in sourcechecks:
   try:
    pp=bind(p,role='current_source');actual=inputs[str(pp)]['sha256'];sourcechecks[key]={'path':str(pp),'expected_sha256':expected,'current_sha256':actual,'status':'matches'if actual==expected else'current_source_drift'}
   except FileNotFoundError:sourcechecks[key]={'path':str(path(p)),'expected_sha256':expected,'current_sha256':None,'status':'current_source_unavailable'}
  return sourcechecks[key]
 def frozen(manifestpath):
  manifest=load(manifestpath,role='profile_manifest');data={}
  for f in manifest['files']:
   compressed=read(f['path'],f['compressed_sha256'],'frozen_profile');raw=gzip.decompress(compressed)
   if hbytes(raw)!=f['uncompressed_sha256']:raise ValueError('Uncompressed frozen profile mismatch '+f['path'])
   if f['original_name']in ['profile_provenance.json','plan.json']:data[f['original_name']]=json.loads(raw)
  return manifest,data
 study=root/'ontology/studies/v100_roofline_537';lock=None
 if not a.offline_snapshot:
  lock=P('/tmp/hecbench-roofline-metadata.lock').open('rb');fcntl.flock(lock,fcntl.LOCK_SH)
 try:
  protocol=load(study/'protocol.json',role='protocol');census=load(protocol['source_census']['path'],role='canonical_census');names=sorted({x['benchmark']for x in census['rows']}|set(census.get('no_aggregate_eligible_accepted_contract_names',[])))
  if len(names)!=537:raise ValueError('Expected canonical537 names')
  retained=load(a.retained_family_audit,role='strict_family_audit')if a.retained_family_audit else None
  admitted={}
  if retained:
   for p,h in retained['input_hashes'].items():
    # This is a generating analysis-tool receipt, not executed benchmark source.
    # Its historical hash stays inside the bound audit; today's shared file may
    # legitimately evolve without altering any measured capture or audit bytes.
    if path(p).resolve()==root/'tools/export_v100_retained_roi.py':continue
    bind(p,h,'strict_family_dependency')
   admitted={(x['benchmark'],x['model']):x for x in retained['records']if x.get('accepted_complete_scientific_kernel_family')is True}
  for b in names:
   lp=study/'benchmarks'/(b+'.json');ledger=load(lp,role='selected_ledger')if lp.exists()else{};contract=ledger.get('accepted_named_contract')or{};dp=root/'ontology/studies/julia_abstractions_537/domains'/(b+'.json');domain=load(dp,role='scientific_domain')if dp.exists()else{};intent=domain.get('scientific_intent')or{};family=intent.get('primary_family')or'unknown'
   for m in MODELS:
    state=(ledger.get('models')or{}).get(m,{})
    slot={'benchmark':b,'model':m,'scientific_family':family,'scientific_task':intent.get('scientific_task'),'scientific_family_review_status':intent.get('review_status'),'domain_ontology_record':str(dp)if dp.exists()else None,'base_ontology_resource_bias_unchanged':True,'configuration_id':contract.get('configuration_id'),'accepted_contract_outcome':contract.get('outcome'),'reference_qualification':contract.get('qualification'),'scope_review':contract.get('scope_review'),'current_primary_mean_seconds':((contract.get('models')or{}).get(m)or{}).get('mean'),'primary_mean_is_separate_unprofiled_five_run_metric':True,'profiling_status':state.get('profiling_status','missing_model_record'),'plan_status':state.get('profile_plan_status'),'plan_gates':state.get('profile_plan_blocking_gates',[]),'coverage':state.get('roi_coverage','unknown'),'coverage_reason':state.get('reason'),'selected_profile_ids':state.get('profile_ids',[]),'latest_attempt':state.get('latest_profile_attempt'),'registry_assessed':True,'selected_valid_kernel_observation_count':None,'complete_elapsed_host_ROI':False,'whole_benchmark_causal_bottleneck':'unknown','per_benchmark_geometric_regime':None,'observation_qualifications':[],'stage_assessments':[],'scientific_family_aggregate':None,'source_current_binding_checks':[],'evidence':{'selected_ledger':str(lp),'selected_ledger_sha256':inputs.get(str(lp),{}).get('sha256')},'issues':[]};slots.append(slot)
    if (b,m)in admitted:
     rr=admitted[b,m];slot['scientific_family_aggregate']={'audit_path':str(a.retained_family_audit),'audit_sha256':inputs[str(a.retained_family_audit.resolve())]['sha256'],'configuration_id':rr['configuration_id'],'source_launch_proof':rr.get('source_launch_proof'),'totals':rr['totals'],'scope':rr.get('work_traffic_scope'),'complete_elapsed_host_ROI':False,'causal_bottleneck':'unknown'}
    if state.get('hardware_evidence'):
     he=state['hardware_evidence'];bind(he['path'],he['sha256'],'hardware_barrier');slot['evidence']['hardware_barrier']=he
    if contract.get('result'):
     try:
      rp=bind(contract['result'],contract['result_sha256'],'original_numerical_result');cp=rp.with_name('config.json');bind(cp,contract['raw_config_sha256'],'original_configuration');slot['evidence'].update(original_result=str(rp),original_result_sha256=contract['result_sha256'],original_config=str(cp),original_config_sha256=contract['raw_config_sha256'])
     except(ValueError,FileNotFoundError)as e:slot['issues'].append('original_receipt_binding:'+str(e))
    if state.get('profiling_status')!='profile_process_completed':
     slot['assessment_category']=('source_reviewed_runtime_without_profiled_device_kernels'if state.get('profiling_status')=='roofline_not_applicable_no_device_kernels'else'hardware_prerequisite_blocked'if state.get('profiling_status')=='native_hardware_blocked'else'no_valid_selected_counter_capture')
     if state.get('profile_evidence_manifest'):
      try:
       mp=state['profile_evidence_manifest'];mf,fd=frozen(mp);pp=fd.get('profile_provenance.json',{});pl=fd.get('plan.json',{})
       slot['evidence'].update(profile_manifest=mp,profile_manifest_sha256=inputs[str(path(mp))]['sha256'],counter_sha256=pp.get('counters_sha256'),failed_or_blocked_process_status=pp.get('status'),failed_or_blocked_process_outcome=pp.get('profile'),source_result_sha256=pl.get('reference_result_sha256'),source_config_sha256=pl.get('raw_config_sha256'))
       for sp,h in pl.get('execution_bindings',{}).items():slot['source_current_binding_checks'].append(sourcecheck(sp,h))
      except(ValueError,KeyError,FileNotFoundError,EOFError)as e:slot['issues'].append('failed_or_blocked_capture_receipt:'+str(e))
     continue
    try:
     manifestpath=state['profile_evidence_manifest'];manifest,fd=frozen(manifestpath);prov=fd['profile_provenance.json'];plan=fd['plan.json'];slot['evidence'].update(profile_manifest=manifestpath,profile_manifest_sha256=inputs[str(path(manifestpath))]['sha256'],source_result_sha256=plan.get('reference_result_sha256'),source_config_sha256=plan.get('raw_config_sha256'),counter_sha256=prov.get('counters_sha256'),collector_sha256=prov.get('collector_sha256'),selector=prov.get('selection_override'),source_binding_recheck_after=prov.get('binding_recheck_after'),device_identity_verified=prov.get('profile_device_identity_verified'),gpu_uuid=prov.get('gpu_uuid'),counter_protocol={'cache':'all_cold_replay','clock_control':'none','replay_mode':'kernel'})
     if prov.get('status')!='profile_process_completed'or prov.get('profile',{}).get('returncode')!=0:raise ValueError('Frozen selected process did not complete')
     files={f['original_name']:f for f in manifest['files']}
     if files['plan.json']['uncompressed_sha256']!=prov.get('plan_sha256'):raise ValueError('Plan/provenance mismatch')
     if files.get('counters.csv',{}).get('uncompressed_sha256')!=prov.get('counters_sha256'):raise ValueError('Counter/provenance mismatch')
     selectedparents={str(path(x).resolve())for x in state.get('profile_ids',[])}
     if str(path(manifestpath).resolve().parent)not in selectedparents:raise ValueError('Manifest not inside declared selected profile')
     actual_provenance_sha=files['profile_provenance.json']['uncompressed_sha256'];slot['evidence']['actual_selected_profile_provenance_sha256']=actual_provenance_sha
     if state.get('profile_provenance_sha256')and state['profile_provenance_sha256']!=actual_provenance_sha:
      slot['evidence']['ledger_recorded_profile_provenance_sha256']=state['profile_provenance_sha256'];slot['issues'].append('stale_ledger_top_level_provenance_SHA_selected_manifest_verified');slot['observation_qualifications'].append('stale_ledger_top_level_provenance_SHA_selected_manifest_verified')
     if plan.get('reference_result_sha256')!=contract.get('result_sha256')or plan.get('raw_config_sha256')!=contract.get('raw_config_sha256'):raise ValueError('Profile and selected numerical contract differ')
     if prov.get('binding_recheck_after')!='unchanged':slot['observation_qualifications'].append('source_before_after_binding_unproved')
     if not prov.get('profile_device_identity_verified'):slot['observation_qualifications'].append('legacy_device_PCI_receipt_absent')
     for p,h in {**plan.get('execution_bindings',{}),**prov.get('additional_current_profile_input_bindings',{})}.items():slot['source_current_binding_checks'].append(sourcecheck(p,h))
     if any(x['status']!='matches'for x in slot['source_current_binding_checks']):slot['observation_qualifications'].append('current_source_reproduction_has_drift_or_unavailable_path')
     link=state['kernel_observations']
     if path(link['path']).resolve().parent!=path(manifestpath).resolve().parent:raise ValueError('Observations not inside selected manifest profile')
     raw=read(link['path'],link['sha256'],'selected_kernel_observations');raw=gzip.decompress(raw)if link.get('compression')=='gzip'else raw
     if link.get('uncompressed_sha256')and hbytes(raw)!=link['uncompressed_sha256']:raise ValueError('Kernel annotation uncompressed mismatch')
     records=json.loads(raw)['records']
     if state.get('sampled_kernel_count')!=len(records):raise ValueError('Selected observation count mismatch')
     seen=set();stagegroups=collections.defaultdict(list);observed=[]
     app=state.get('dram_roofline_applicability')or'source_tier_unreviewed_HBM_projection_only'
     for k in records:
      ctx=k.get('scope_context')or{};ident=k['identity'];identity=tuple(str(ident.get(x))for x in ['ID','Process ID','Context','Stream','Kernel Name'])
      if identity in seen:raise ValueError('Duplicate selected kernel identity')
      seen.add(identity)
      if ctx.get('configuration_id')!=plan.get('configuration_id')or ctx.get('reference_result_sha256')!=plan.get('reference_result_sha256'):raise ValueError('Kernel annotation/config/result mismatch')
      tierapp=k.get('dram_roofline_applicability')or ctx.get('dram_roofline_applicability')or app
      # Zerocopy's accepted contract explicitly specifies Host mapped primary data.
      if b=='zerocopy':tierapp='not_applicable_host_mapped_primary_data_requires_PCIe_tier'
      stage=ctx.get('ROI_stage')or'unassigned';ph=phase(ctx);pr=projections(k,tierapp)
      kr={'benchmark':b,'model':m,'configuration_id':plan.get('configuration_id'),'identity':ident,'semantic_stage':stage,'stage_membership':ctx.get('semantic_family_in_primary_ROI'),'sample_phase':ctx.get('sample_phase'),'phase_qualification':ph,'precision_class':k.get('precision_class','unknown'),'precision_detection':k.get('precision_detection'),'DRAM_applicability':tierapp,'scalar_work_is_not_total_cost':True,'duration_seconds':k.get('duration_seconds'),'dram_bytes':k.get('dram_bytes'),'L2_transaction_bytes':k.get('L2_transaction_bytes'),'projections':pr,'observation_sha256':hbytes(json.dumps(k,sort_keys=True).encode()),'complete_elapsed_host_ROI':False,'causal_bottleneck':'unknown'};observed.append(kr);stagegroups[(stage,ph)].append(kr)
     slot['selected_valid_kernel_observation_count']=len(observed);slot['assessment_category']='process_completed_no_kernel_observations'if not observed else 'valid_selected_observations_no_counted_FP'if all(x['precision_class']=='no_counted_FP'for x in observed)else'valid_selected_observations';slot['precision_class_distribution']=counts(x['precision_class']for x in observed);slot['phase_qualification_distribution']=counts(x['phase_qualification']for x in observed);slot['memory_domain_qualification_distribution']=counts(x['DRAM_applicability']for x in observed)
     for(stage,ph),ks in sorted(stagegroups.items()):
      projections_by_key=collections.defaultdict(list)
      for k in ks:
       for p in k['projections']:projections_by_key[p['precision'],p['memory_tier']].append(p)
      sa={'semantic_stage_label':stage,'phase_qualification':ph,'source_stage_is_proved':ph not in ['source_membership_unassigned_or_candidate','explicit_source_exclusion'],'kernel_observation_count':len(ks),'precision_class_distribution':counts(x['precision_class']for x in ks),'observed_kernel_names':sorted({x['identity']['Kernel Name']for x in ks}),'launch_geometries':sorted({str(x['identity'].get('Grid Size'))+' / '+str(x['identity'].get('Block Size'))for x in ks}),'precision_tier_positioning':[],'whole_elapsed_ROI_claim':False,'causal_bottleneck':'unknown'}
      for(precision,tier),ps in sorted(projections_by_key.items()):
       valid=[p for p in ps if p['positioning_admitted']];p={'precision':precision,'memory_tier':tier,'projection_observation_count':len(ps),'qualified_positioning_count':len(valid),'geometric_regime_distribution':counts(x['geometric_regime']for x in valid),'nonpositioning_reason_distribution':counts(x['applicability_reason']for x in ps if not x['positioning_admitted']),'intensity_range':extrema(x['intensity']for x in valid),'normalized_intensity_over_ridge_range':extrema(x['intensity_over_ridge']for x in valid),'kernel_replay_rate_range':extrema(x['achieved_kernel_replay_flops_per_second']for x in valid),'range_not_mean':True};sa['precision_tier_positioning'].append(p);stagecsv.append({'benchmark':b,'model':m,'scientific_family':family,'stage':stage,'phase_qualification':ph,'precision':precision,'memory_tier':tier,'kernel_observations':len(ks),'qualified_positions':len(valid),'memory_side':p['geometric_regime_distribution'].get('memory_side',0),'near_ridge':p['geometric_regime_distribution'].get('near_ridge',0),'compute_side':p['geometric_regime_distribution'].get('compute_side',0),'intensity_min':p['intensity_range']['min'],'intensity_max':p['intensity_range']['max'],'normalized_I_min':p['normalized_intensity_over_ridge_range']['min'],'normalized_I_max':p['normalized_intensity_over_ridge_range']['max']})
      slot['stage_assessments'].append(sa)
     family_candidate=bool(observed)and all(k['phase_qualification']=='source_counted_retained_invocation_not_whole_elapsed_ROI'for k in observed)and prov.get('retained_window_expected_count_confirmed')is True and prov.get('selection_expected_launch_count_matches')is True
     slot['additional_family_aggregation_review']={'source_counted_complete_family_candidate':family_candidate,'strict_aggregate_admitted':slot['scientific_family_aggregate']is not None,'candidate_basis':'All selected scientific samples source-proven retained invocation, expected actual launch count confirmed; still requires independent strict counter/source/schema admission.'if family_candidate else'No complete source-counted family proof; selected source candidates/first-geometry samples must not be extrapolated.','whole_elapsed_ROI_claim':False}
     kernelrows.extend(observed)
    except(ValueError,KeyError,FileNotFoundError,EOFError)as e:
     slot['assessment_category']='selected_capture_evidence_invalid_or_incomplete';slot['issues'].append(str(e));slot['selected_valid_kernel_observation_count']=None;slot['stage_assessments']=[]
  bykey={(s['benchmark'],s['model']):s for s in slots}
  for b in names:
   for m in MODELS[1:]:
    n=bykey[b,'cuda'];t=bykey[b,m];nf=n.get('scientific_family_aggregate');tf=t.get('scientific_family_aggregate');reason=None
    if not nf or not tf:reason='complete_source_counted_compatible_family_evidence_missing'
    elif n.get('selected_valid_kernel_observation_count')is None or t.get('selected_valid_kernel_observation_count')is None:reason='selected_valid_observations_missing'
    elif any('drift'in q or'unproved'in q for q in n['observation_qualifications']+t['observation_qualifications']):reason='current_source_or_capture_binding_requires_review'
    elif nf['configuration_id']!=tf['configuration_id']or nf['configuration_id']!=n['configuration_id']or t['configuration_id']!=n['configuration_id']:reason='configuration_mismatch'
    elif nf['source_launch_proof']['source_ROI']!=tf['source_launch_proof']['source_ROI']:reason='source_ROI_sequence_mismatch'
    elif nf['totals']['actual_scientific_launches']!=tf['totals']['actual_scientific_launches']:reason='launch_multiplicity_mismatch'
    else:
     na=admitted[b,'cuda'];ta=admitted[b,m]
     if na['capture_provenance']['counter_sha256']!=n['evidence'].get('counter_sha256')or ta['capture_provenance']['counter_sha256']!=t['evidence'].get('counter_sha256'):reason='family_audit_not_same_selected_capture'
    if reason:pairstats[reason]+=1;continue
    nF=nf['totals'];tF=tf['totals'];pairs.append({'benchmark':b,'target_model':m,'configuration_id':n['configuration_id'],'comparison_scope':'source_counted_selected_scientific_family_cold_replay_not_elapsedROI','precision':'fp32','native_totals':nF,'target_totals':tF,'DRAM_integrated_intensity_ratio_target_over_native':tF['fp32_flop_per_DRAM_byte']/nF['fp32_flop_per_DRAM_byte']if nF['fp32_flop_per_DRAM_byte']else None,'cold_replay_summed_device_duration_ratio_target_over_native':tF['sum_device_kernel_replay_seconds']/nF['sum_device_kernel_replay_seconds'],'primary_mean_speedup_joined_to_kernel_samples':None,'causal_bottleneck':'unknown','host_ROI_performance_claim':False});pairstats['admitted_source_counted_family_pair']+=1
  bind(P(__file__).resolve(),role='assessment_exporter')
  for p,info in inputs.items():
   if sha(p)!=info['sha256']:raise ValueError('Input changed during assessment '+p)
 finally:
  if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 stats={m:{'denominator_names':537,'profiling_status_distribution':counts(s['profiling_status']for s in slots if s['model']==m),'assessment_category_distribution':counts(s['assessment_category']for s in slots if s['model']==m),'ROI_coverage_distribution':counts(s['coverage']for s in slots if s['model']==m),'valid_selected_kernel_observation_count':sum(s['selected_valid_kernel_observation_count']or 0 for s in slots if s['model']==m),'names_with_valid_selected_observations':sum((s['selected_valid_kernel_observation_count']or 0)>0 for s in slots if s['model']==m),'complete_scientific_family_aggregate_count':sum(s['scientific_family_aggregate']is not None for s in slots if s['model']==m),'complete_elapsed_host_ROI_positions':0,'causal_bottleneck_assignments':0}for m in MODELS}
 familycoverage=collections.Counter((s['scientific_family'],s['model'],s['assessment_category'])for s in slots)
 out.mkdir(parents=True);payload={'schema_version':1,'denominators':{'benchmark_names':537,'models':list(MODELS),'model_slots':2148,'Julia_implementation_slots':1611},'registry_assessment_complete':len(slots)==2148,'complete_GPU_ROI_campaign_claim':False,'causal_bottleneck_campaign_claim':False,'snapshot_mode':'quiescent_offline_rehash'if a.offline_snapshot else'shared_metadata_lock_and_full_input_rehash','input_bindings':inputs,'per_model':stats,'model_assessments':slots,'compatible_native_family_comparisons':pairs,'native_comparison_admission_counts':dict(pairstats),'correlations':{'intensity_speedup':None,'domain_performance':None,'reason':'Too few explicitly compatible complete-family controls; partial-stage samples cannot be joined to whole primary performance as causal or benchmark aggregates.'},'additional_family_aggregation_review':{'already_strictly_source_counted_benchmarks':sorted({b for b,m in admitted}),'strict_admitted_model_families':len(admitted),'source_counted_not_yet_strictly_admitted_candidates':[{'benchmark':s['benchmark'],'model':s['model'],'review':s['additional_family_aggregation_review']}for s in slots if s.get('additional_family_aggregation_review',{}).get('source_counted_complete_family_candidate')and not s['scientific_family_aggregate']],'other_finished_processes':'Source candidate only; require complete source stage/launch ledger + retained selector + numerical checks + exact actual match count, never multiply partial samples.','retained_audit':str(a.retained_family_audit)if a.retained_family_audit else None,'historical_generating_tool_receipts':{p:h for p,h in retained.get('input_hashes',{}).items()if path(p).resolve()==root/'tools/export_v100_retained_roi.py'}if retained else{},'historical_tool_note':'Generating analysis-tool version hashes are retained in the immutable bound audit; current shared analysis scripts are not executed benchmark sources and are not required to remain at historical bytes.'},'scientific_family_coverage':[{'family':f,'model':m,'assessment_category':c,'benchmark_names':n}for(f,m,c),n in sorted(familycoverage.items())],'limitations':['Per-kernel geometric positioning only, not causal bottleneck.','Full registry assessment is not full GPU ROI coverage.','No equal-weight kernel averages or missing-as-zero positions.','Primary five-run means remain separate from profiler counter times.','Original artifact bytes not re-audited by this assessment.']};dump(out/'assessment.json',payload);dump(out/'descriptor_assessment.json',compact_descriptors(payload));dump(out/'kernel_stage_observations.json',{'schema_version':1,'records':kernelrows,'count_unit':'selected valid kernel observations, not benchmark names'});(out/'PROTOCOL.md').write_text(PROTOCOL)
 with(out/'model_assessment.csv').open('w',newline='')as f:
  fields=['benchmark','model','scientific_family','configuration_id','accepted_contract_outcome','reference_qualification','current_primary_mean_seconds','profiling_status','assessment_category','coverage','selected_valid_kernel_observation_count','complete_elapsed_host_ROI','whole_benchmark_causal_bottleneck'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:s.get(k)for k in fields}for s in slots)
 with(out/'stage_positioning.csv').open('w',newline='')as f:
  fields=['benchmark','model','scientific_family','stage','phase_qualification','precision','memory_tier','kernel_observations','qualified_positions','memory_side','near_ridge','compute_side','intensity_min','intensity_max','normalized_I_min','normalized_I_max'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(stagecsv)
 dump(out/'INTERFACE.json',{'entrypoint':'tools/export_v100_benchmark_assessment.py','command':'python3 tools/export_v100_benchmark_assessment.py --output-root /tmp/FRESH --retained-family-audit /tmp/FAMILY/audit.json','writes':'fresh /tmp only; never replaces outputs','live_consistency':'shared coordinator metadata flock then rehash every input before unlock','offline':'--offline-snapshot only for explicitly quiescent relocated artifact','models':'537 names x cuda/cuda_jl/ka/jacc','payloads':['descriptor_assessment.json: compact evidence-scope descriptor for every slot','assessment.json: 2148 assessments and compatible-family-only comparisons','kernel_stage_observations.json: measured selected valid stages with scalar/tensor/tier projections','model_assessment.csv','stage_positioning.csv','PROTOCOL.md'],'nonclaims':['No full elapsedROI or causal classifications','No primary means changed','No equal-kernel benchmark averages','No missing positions converted to zero']})
 print(json.dumps({'assessment':str(out/'assessment.json'),'model_slots':len(slots),'valid_selected_kernel_observations':len(kernelrows),'compatible_native_family_pairs':len(pairs),'categories':counts(s['assessment_category']for s in slots)}))
if __name__=='__main__':main()
