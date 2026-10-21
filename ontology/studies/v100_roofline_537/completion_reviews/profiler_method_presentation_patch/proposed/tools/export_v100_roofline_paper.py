#!/usr/bin/env python3
"""Deterministic compact V100 campaign publication snapshot; CPU/read-only inputs.

Writes only a fresh explicit /tmp directory. No raw counters/captures, manuscript,
shared ledgers, GPU commands, or benchmark-level averaging. All537x4 slots remain.
"""
import argparse,collections,contextlib,csv,fcntl,hashlib,io,json,math,pathlib,sys
sys.dont_write_bytecode=True
csv.field_size_limit(64*1024*1024)
P=pathlib.Path;MODELS=('cuda','cuda_jl','ka','jacc');LABELS={'cuda':'CUDA','cuda_jl':'CUDA.jl','ka':'KA','jacc':'JACC'}
COMPLETE='profile_process_completed';REGIMES={'memory_side','near_ridge','compute_side'}
def sha(b):return hashlib.sha256(b).hexdigest()
def num(x):
 try:v=float(x)
 except(TypeError,ValueError):return None
 return v if math.isfinite(v)else None
def yes(x):return x is True or str(x).lower()=='true'
def phase(r):
 semantic=r.get('semantic_family_in_primary_ROI');sample=r.get('sample_phase')or'unknown';stage=r.get('stage')or'unassigned'
 if semantic is False or str(semantic).lower()=='false':return 'source_excluded_setup_or_audit'
 if stage=='unassigned'or stage.startswith('candidate:')or semantic in [None,'']:return 'ROI_unassigned_or_candidate'
 if 'warmup'in sample or'first_'in sample:return 'first_invocation_or_warmup_qualified'
 if 'retained'in sample:return 'retained_primary_sample_not_complete_phase'
 return 'source_family_membership_only_phase_unproved'

def profiler_identity(record):
    """Descriptive identity only; never promote unknown identity to calibration approval."""
    declared=record.get('profiler_method') or {}
    if not isinstance(declared,dict): declared={}
    method=record.get('profiler_method_id') or declared.get('method_id')
    if not method or method=='legacy2022_recorded_or_identity_incomplete':
        method='unknown_profiler_method'
    return {'profiler_method_id':method,
            'profiler_version':record.get('profiler_version') or 'unknown_profiler_version',
            'method_review_sha256':record.get('method_review_sha256') or declared.get('review_sha256') or 'unknown_method_review'}
def profiler_stratum(record):
    identity=profiler_identity(record)
    return tuple(identity[k] for k in ('profiler_method_id','profiler_version','method_review_sha256'))

def counter_dict(c):return dict(sorted(c.items()))
def tex(s):
 return str(s).replace('\\','\\textbackslash{}').replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('#',r'\#')
def write_csv(p,rows,fields):
 with p.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--usage-receipt',type=P);ap.add_argument('--offline-snapshot',action='store_true',help='Explicit quiescent snapshot mode for relocated artifact; rehash all inputs after reading, no shared lock creation.');a=ap.parse_args();root=a.source_root.resolve();out=a.output_root.absolute()
 if not out.resolve().is_relative_to(P('/tmp'))or out.resolve()==P('/tmp')or out.exists()or out.is_symlink():ap.error('Fresh output root strictly under /tmp required')
 study=root/'ontology/studies/v100_roofline_537';inputs={}
 def read(p):
  p=P(p);b=p.read_bytes();key=str(p.relative_to(root))if p.is_relative_to(root)else str(p);inputs[key]={'sha256':sha(b),'bytes':len(b)};return b
 def j(p):return json.loads(read(p))
 def table(p):return list(csv.DictReader(io.StringIO(read(p).decode())))
 lock=P('/tmp/hecbench-roofline-metadata.lock')
 if not a.offline_snapshot and not lock.exists():raise ValueError('Coordinator metadata lock missing; use --offline-snapshot only for a quiescent artifact')
 with (contextlib.nullcontext(None) if a.offline_snapshot else lock.open('r'))as f:
  if f is not None:fcntl.flock(f,fcntl.LOCK_SH)
  protocol=j(study/'protocol.json');progress=j(study/'PROGRESS.json')
  census_path=root/protocol['source_census']['path'];census=j(census_path)
  names=sorted({r['benchmark']for r in census['rows']}|set(census.get('no_aggregate_eligible_accepted_contract_names',[])))
  if len(names)!=537:raise ValueError('Canonical name union must be537, got '+str(len(names)))
  ledgers={};domains={}
  for name in names:
   p=study/'benchmarks'/(name+'.json');ledgers[name]=j(p)if p.exists()else{}
   p=root/'ontology/studies/julia_abstractions_537/domains'/(name+'.json');domains[name]=j(p)if p.exists()else{}
  kernel_rows=table(study/'kernel_audit.csv');scalar_rows=table(study/'kernel_roofline_points.csv');hier_rows=table(study/'kernel_hierarchical_points.csv')
  if(study/'kernel_tensor_roofline_points.csv').is_file():hier_rows+=table(study/'kernel_tensor_roofline_points.csv')
  # Record compact model export as cross-reference, but use all537 ledgers for denominator authority.
  if(study/'model_audit.csv').exists():read(study/'model_audit.csv')
  receipt=j(a.usage_receipt.resolve())if a.usage_receipt else None
  if a.offline_snapshot:
   for name,binding in inputs.items():
    p=P(name);p=p if p.is_absolute()else root/p
    if sha(p.read_bytes())!=binding['sha256']:raise ValueError('Offline snapshot changed during export: '+name)
  if f is not None:fcntl.flock(f,fcntl.LOCK_UN)
 stats={m:{'denominator_names':537,'status_counts':collections.Counter(),'ROI_coverage_counts':collections.Counter(),'contract_outcomes':collections.Counter(),'precision_classes_in_completed_samples':collections.Counter(),'completed_kernel_observations':0,'failed_process_kernel_rows_excluded':0,'whole_ROI_classifications':0,'causal_bottleneck_classifications':0,'unassigned_or_candidate_kernel_observations':0,'first_or_warmup_qualified_kernel_observations':0,'sampled_phase_counts':collections.Counter(),'recorded_sample_phase_labels':collections.Counter(),'causal_labels_without_explicit_evidence':0,'profiler_stratum_completed_sample_counts':collections.Counter(),'DRAM_tier_applicability_counts':collections.Counter()}for m in MODELS}
 slotrows=[];familycounts=collections.Counter();familyrecords={};slots={}
 for name in names:
  ledger=ledgers[name];contract=ledger.get('accepted_named_contract')or{};intent=domains[name].get('scientific_intent',{})
  family=intent.get('primary_family')or'unknown';familyrecords[name]={'family':family,'operator':intent.get('scientific_task')or intent.get('mathematical_operation')or intent.get('operator'),'review_status':intent.get('review_status'),'domain_record':'ontology/studies/julia_abstractions_537/domains/'+name+'.json'if domains[name]else None}
  for m in MODELS:
   state=ledger.get('models',{}).get(m,{})
   status=state.get('profiling_status','missing_model_record');coverage=state.get('roi_coverage','unknown');st=stats[m];st['status_counts'][status]+=1;st['ROI_coverage_counts'][coverage]+=1;st['contract_outcomes'][contract.get('outcome')or('hardware_unverified'if status=='native_hardware_blocked'else'unknown')]+=1
   causal=state.get('observed_bottleneck')or state.get('causal_bottleneck');causal_known=causal not in [None,'','unknown','unknown_not_aggregated'];whole=state.get('roofline_region');whole_proved=bool(whole) and state.get('full_timed_ROI_coverage_confirmed')is True and status==COMPLETE;causal_proved=causal_known and bool(state.get('causal_evidence')or state.get('observed_bottleneck_evidence'))and status==COMPLETE;st['whole_ROI_classifications']+=whole_proved;st['causal_bottleneck_classifications']+=causal_proved;st['causal_labels_without_explicit_evidence']+=causal_known and not causal_proved
   key=(name,m);slots[key]=state;familycounts[(family,m,status)]+=1
   mean=((contract.get('models')or{}).get(m)or{}).get('mean')
   slotrows.append({**profiler_identity(state),'benchmark':name,'model':m,'scientific_family':family,'configuration_id':contract.get('configuration_id'),'profiling_status':status,'ROI_coverage':coverage,'recorded_kernel_observation_count':state.get('sampled_kernel_count'),'whole_ROI_geometric_classification':whole if whole_proved else None,'recorded_causal_label':causal,'actual_causal_bottleneck':causal if causal_proved else None,'contract_outcome':contract.get('outcome'),'reference_qualification':contract.get('qualification'),'primary_unprofiled_mean_seconds':mean,'kernel_evidence_manifest':state.get('profile_evidence_manifest'),'ontology_record':'ontology/studies/v100_roofline_537/benchmarks/'+name+'.json'})
 for r in kernel_rows:
  m=r['model'];key=(r['benchmark'],m)
  if m not in stats or key not in slots:raise ValueError('Kernel outside canonical slot universe')
  if r.get('profiling_status')!=COMPLETE or slots[key].get('profiling_status')!=COMPLETE:
   stats[m]['failed_process_kernel_rows_excluded']+=1;continue
  st=stats[m];st['profiler_stratum_completed_sample_counts'][json.dumps(profiler_stratum(r))]+=1;st['completed_kernel_observations']+=1;st['precision_classes_in_completed_samples'][r.get('precision_class')or'unknown']+=1;st['DRAM_tier_applicability_counts'][r.get('dram_roofline_applicability')or slots[key].get('dram_roofline_applicability')or 'source_tier_unreviewed_HBM_projection_only']+=1;ph=phase(r);st['sampled_phase_counts'][ph]+=1;st['unassigned_or_candidate_kernel_observations']+=ph=='ROI_unassigned_or_candidate';st['first_or_warmup_qualified_kernel_observations']+=('warmup'in(r.get('sample_phase')or'')or'first_'in(r.get('sample_phase')or''));st['recorded_sample_phase_labels'][r.get('sample_phase')or'unknown']+=1
 predictions=collections.Counter();excluded=collections.Counter()
 # Hierarchical export includes DRAM andL2 exactly once; never add scalar andhierarchical CSVs.
 for r in hier_rows:
  key=(r['benchmark'],r['model']);m=r['model']
  if key not in slots:raise ValueError('Projection outside canonical slots')
  if r.get('profiling_status')!=COMPLETE or slots[key].get('profiling_status')!=COMPLETE:excluded['failed_or_noncompleted_process']+=1;continue
  applicability=r.get('dram_roofline_applicability')or slots[key].get('dram_roofline_applicability')or 'source_tier_unreviewed_HBM_projection_only'
  if applicability.startswith('not_applicable'):
   excluded['source_memory_tier_not_applicable_retained_as_raw_diagnostic']+=1;continue
  work=num(r.get('executed_flops'))
  if work is None or work<=0:excluded['missing_or_nonpositive_counted_work']+=1;continue
  if r.get('roofline_regime')not in REGIMES:excluded['undefined_or_missing_envelope']+=1;continue
  if num(r.get('intensity_over_ridge'))is None:excluded['missing_normalized_intensity']+=1;continue
  family=familyrecords[r['benchmark']]['family'];predictions[(*profiler_stratum(r),m,r.get('precision')or'unknown',r.get('memory_tier')or'unknown',r['roofline_regime'],phase(r),family,applicability)]+=1
 # Correlations are deliberately gated on explicit complete comparable phase proofs,
 # not kernel-name matches, sourcefamily inclusion or first/retained samples.
 rejection=collections.Counter();pairs=[]
 for name in names:
  for m in MODELS[1:]:
   native=slots[(name,'cuda')];other=slots[(name,m)]
   if native.get('profiling_status')!=COMPLETE or other.get('profiling_status')!=COMPLETE:rejection['both_processes_not_completed']+=1;continue
   ns,ts=profiler_stratum(native),profiler_stratum(other)
   if any(value.startswith('unknown_')for value in ns+ts):rejection['profiler_method_identity_incomplete']+=1;continue
   if ns!=ts:rejection['profiler_method_version_or_review_mismatch']+=1;continue
   if not(native.get('full_timed_ROI_coverage_confirmed')is True and other.get('full_timed_ROI_coverage_confirmed')is True):rejection['complete_timed_phase_coverage_unproved']+=1;continue
   proof=other.get('native_comparison_proof')or{}
   need=['matched_source_ROI','matched_cases_and_actual_inputs','matched_arithmetic_precision','same_device','complete_phase_coverage','aggregation_work_bytes_time_compatible']
   if proof.get('review_status')!='approved'or not all(proof.get(k)is True for k in need)or not proof.get('evidence_sha256'):rejection['explicit_matched_comparison_proof_missing']+=1;continue
   # Publication of future eligible statistics requires a separately reviewed
   # full-ROI aggregate with no equal-kernel averaging; retain candidates only.
   pairs.append({**profiler_identity(other),'benchmark':name,'model':m,'proof':proof,'analysis_status':'eligible_proof_requires_reviewed_aggregate_export'})
 normalized=[]
 for m in MODELS:
  for k,v in list(stats[m].items()):
   if isinstance(v,collections.Counter):stats[m][k]=counter_dict(v)
 familyrows=[{'scientific_family':f,'model':m,'profiling_status':s,'names':n}for(f,m,s),n in sorted(familycounts.items())]
 predrows=[{'profiler_method_id':method,'profiler_version':version,'method_review_sha256':review,'model':m,'precision':p,'memory_tier':tier,'geometric_regime':regime,'sample_phase_qualification':ph,'scientific_family':family,'memory_tier_applicability':applicability,'kernel_projection_count':n}for(method,version,review,m,p,tier,regime,ph,family,applicability),n in sorted(predictions.items())]
 payload={'schema_version':1,'study':'v100_roofline_537','campaign_status':'incomplete_profiling_campaign','snapshot_consistency':'explicit_quiescent_snapshot_with_postread_hash_check' if a.offline_snapshot else 'coordinator_shared_metadata_lock','source_progress_updated_at':progress.get('updated_at'),'denominators':{'benchmark_names':537,'models':list(MODELS),'model_slots':2148,'native_baseline':'cuda','Julia_implementation_slots':1611,'kernel_projection_counts_are_not_benchmark_counts':True},'per_model':stats,'geometric_prediction_counts':predrows,'excluded_projection_rows':counter_dict(excluded),'ontology_families':familyrecords,'native_comparison_analysis':{'status':'insufficient_complete_matched_phase_and_aggregate_evidence','candidate_benchmark_model_pairs':537*3,'proof_candidate_count':len(pairs),'proof_candidates':pairs,'rejection_counts':counter_dict(rejection),'numerical_speedup_correlation':None,'normalized_intensity_ridge_correlation':None,'occupancy_correlation':None,'why_null':'No kernel-weighted benchmark average or correlation inferred from incomplete/partial samples; explicit full compatible aggregate export is required even if a future proof candidate exists.'},'method':{'work':'Executed precision-specific scalar/tensor projections; useful-source triad historical plot remains separate.','prediction_scope':'Successful process kernel samples only; memory_side/compute_side are geometric predictions, never actual bottleneck labels. Unassigned/setup/warmup phases remain separate.','model_status_scope':'Every canonical name/model retained, including hardwarebarriers/failed/pending/blocked/missing.','memory_applicability':'Explicitnot_applicable hostmapped/PCIe-primary tiers excluded from all compute/memory prediction summaries, raw projections retained in source diagnostics. Unreviewed tiers remain qualifiedHBMprojections.','retained_selectors':'Current trial2/sourceordinal proof is a fresh profiling selection, not historicalprimaryexecution or completeelapsedhostROI.','precision_scope':'Scalar classes andtensor remain separate; mixed classes are projections not summed arithmetic/peaks.','timing':'Primaryunprofiled mean2–5 remains original configuration value; replay duration never replaces it.','ontology':'Source-reviewed scientificfamily/operator fields only; catalogue inferredresource_bias untouched.','aggregation':'Counts of projection observations only, not equal-kernel benchmark means or causalclaims.','raw_audit':'InputSHA bindings reproduce this compact snapshot, not an independent recertification of all raw counter values.'},'input_bindings':dict(sorted(inputs.items())),'input_digest':sha(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()),'exporter_sha256':sha(P(__file__).read_bytes()),'shared_analysis_usage_receipt':receipt}
 out.mkdir(parents=True);(out/'campaign_snapshot.json').write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n')
 write_csv(out/'model_status.csv',slotrows,list(slotrows[0]));write_csv(out/'family_coverage.csv',familyrows,['scientific_family','model','profiling_status','names']);write_csv(out/'kernel_projection_counts.csv',predrows,['profiler_method_id','profiler_version','method_review_sha256','model','precision','memory_tier','geometric_regime','sample_phase_qualification','scientific_family','memory_tier_applicability','kernel_projection_count'])
 rows=[]
 for label,values in [('Completed profiling processes',[stats[m]['status_counts'].get(COMPLETE,0)for m in MODELS]),('Failed profiling processes',[stats[m]['status_counts'].get('profile_process_failed',0)for m in MODELS]),('Native hardware barriers',[stats[m]['status_counts'].get('native_hardware_blocked',0)for m in MODELS]),('Pending or plan/method blocked',[537-sum(stats[m]['status_counts'].get(k,0)for k in [COMPLETE,'profile_process_failed','native_hardware_blocked'])for m in MODELS]),('Kernel samples (completed processes)',[stats[m]['completed_kernel_observations']for m in MODELS]),('Whole-ROI geometric classifications',[stats[m]['whole_ROI_classifications']for m in MODELS]),('Causal bottleneck classifications',[stats[m]['causal_bottleneck_classifications']for m in MODELS])]:rows.append(tex(label)+' & '+' & '.join(str(x)for x in values)+r'\\')
 latex='\\begin{table}[t]\\centering\\small\n\\caption{Incomplete V100 profiling snapshot. Each model retains all 537 names; samples and geometric predictions are not whole-benchmark or causal classifications.}\\label{tab:v100-campaign-progress}\n\\begin{tabular}{lrrrr}\\toprule\nAssessment & CUDA & CUDA.jl & KA & JACC\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n'
 (out/'campaign_summary.tex').write_text(latex);(out/'README.md').write_text('Deterministic compact publication snapshot, not raw research. Reproduce with tools/export_v100_roofline_paper.py --output-root /tmp/NEW_FRESH_EXPORT. Live campaign exports hold the coordinator metadata lock; relocated quiescent artifacts may explicitly use --offline-snapshot with post-read input rehash. No lock file is created. AllinputSHA bindings and exporterSHA are in campaign_snapshot.json. Usage receipt may be supplied explicitly; post-END receipt generation is not backfilled as measured effort. Failed processes never contribute geometric predictions; partial/unassigned/warmup/setup observations remain qualified. No kernel averages or unsupported correlations. Primary timings and frozen historical plots remain unchanged.\n')
 print(json.dumps({'output_root':str(out),'names':537,'model_slots':2148,'input_files':len(inputs),'completed_kernel_samples':sum(s['completed_kernel_observations']for s in stats.values()),'eligible_complete_comparison_proof_candidates':len(pairs),'prediction_count_rows':len(predrows),'bytes':sum(p.stat().st_size for p in out.iterdir())}))
if __name__=='__main__':main()
