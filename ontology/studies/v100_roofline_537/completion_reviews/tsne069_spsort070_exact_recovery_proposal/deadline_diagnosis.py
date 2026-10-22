import json,gzip,hashlib,re,datetime
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;tri=Path('/tmp/roofline-completion-20261021/final_v19_remaining_gap_triage/triage_all114.json');h=lambda b:hashlib.sha256(b).hexdigest()
def info(p):return {'path':str(p),'sha256':h(p.read_bytes()),'bytes':p.stat().st_size}
records=[]
for row in json.loads(tri.read_bytes())['slots']:
 n,m=row['benchmark'],row['model']
 if n not in ['dispatch','spsort','slu']or(n=='slu'and m!='ka'):continue
 p=W/row['latest_profile_id'];raw=gzip.decompress((p/'execution.log.gz').read_bytes());text=raw.decode();prov=json.loads(gzip.decompress((p/'profile_provenance.json.gz').read_bytes()));plan=json.loads(gzip.decompress((p/'plan.json.gz').read_bytes()));rs=[];partial=[]
 for line in text.splitlines():
  if line.startswith('STUDY_TRIAL '):
   try:rs.append(json.loads(line[len('STUDY_TRIAL '):]))
   except json.JSONDecodeError:partial.append({'bytes':len(line),'sha256':h(line.encode()),'role':'truncated row not counted as a complete trial'})
 assert all(x['correct']is True for x in rs);assert prov['profile']['stop_reason']=='wall_timeout';assert prov.get('profiler_method')is None
 e={'benchmark':n,'model':m,'capture_id':row['latest_profile_id'],'frozen_execution_log':info(p/'execution.log.gz'),'raw_execution_log_sha256':h(raw),'frozen_plan':info(p/'plan.json.gz'),'frozen_provenance':info(p/'profile_provenance.json.gz'),'observed_complete_trials':rs,'partial_rows_not_counted':partial,'deadline':prov['operational_deadline'],'method':{'recorded_profiler_version':prov['profiler_version'],'historical_method_identity':None,'profiler_executable':prov['profiler_executable'],'profiler_executable_sha256':prov['profiler_executable_sha256']},'selector':prov.get('selection_override'),'no_causal_or_complete_ROI_claim':True}
 if n=='spsort':
  assert[x['trial']for x in rs]==[1,2];assert all(x['checked_values']==484725 and x['output_artifact']['count']==969258 and x['mismatch_output_words']==0 for x in rs);der=json.loads(gzip.decompress((p/'derived.json.gz').read_bytes()));e.update(proposal='One exact041-source/selector/method replay at explicit3600processseconds, full3cases*1000statefulCSRSORT+Gather preserved,5checkedtrials required; batch070 plans contain exactall4selectorunion.',counter_rows=len(der['records']),source_complete_batch_seconds_from_declared_normalization=[x['time_seconds']*1000 for x in rs],storage_primary5bytes=969258*4*5,triage_text_correction='Earlier no-complete-trial wording contradicted by latest041 log:2fullcorrecttrialrows, exactoutputcount/mismatchchecks. Priortriage remainsunchanged; this dated note supersedes only that interpretation.')
 elif n=='dispatch':
  assert all(x['source_empty_kernel']and x['execution_checked']and x['checked_values']==0 and x['completed_source_loop_launch_calls']==1102200 for x in rs);e.update(proposal='Ifrootrequires completed observer capture, one fresh exact021-source/per-function-selector legacy2022 traversal with explicit3600processseconds and all5executioncheckedtrials. Do not shorten1100samples/1000batch, fabricate numericoutputs or promoteFPpoint. No launchable dispatchcontroller authorized by this report.',scientific_scope='Emptykernel/API dispatch timing; execution-only outcome, zero useful numerical work. Normalizedperlaunch ROI isnotfullprocess/completebatchwallforecast.',source_calls_per_trial=1102200,source_calls_full5=5511000,source_external_warmup_calls=1)
 else:
  assert[x['trial']for x in rs]==[1,2,3,4];warm='FullSLU warmup correct=1'in text;case0='SLU independent case=0 n=99340 pass=1'in text;assert warm and case0;e.update(proposal='One future fresh018bounded-per-function-selector legacy2022 replay with explicit900processseconds, full5checkedtrials/all4cases+fullASICwarmup retained. Requires separately prepared exact buildmanifest prefix/SHAinverse to prevent historicaloutputs, before anylaunch. Notold007geometrycensus andnotreducedcasecopy.',full_ASIC_warmup_independent_pass_observed=True,source_case0_workspace_uncompressed_bytes=10353*99340*4,source_workspace_capture_chunk_bytes=8*1024**2,source_workspace_encoding='FullactualD2H8MiBchunks streamedSHA256+zlib_level1,zero-check; never skip4GBworkspace.',source_cause_unknown='Fourcompletechecks proveprogress,notaprofilerdeadlock/cause diagnosis. Original300s operationaldeadline includesCPU symbolic/preprocessing/oracle/fullworkspaceaudit outside reportedkernelROI.',build_manifest_path_and_SHA=plan['build_command'][plan['build_command'].index('--build-manifest')+1:])
 records.append(e)
assert len(records)==9
sources=[]
for rel in ['src/dispatch-julia/study_native.cu','src/dispatch-julia/study_common.jl','src/spsort-julia/study_native.cu','src/spsort-julia/study_common.jl','src/slu-julia/full_study.cu']:
 p=W/rel;sources.append({**info(p),'full_text':p.read_text()})
(O/'deadline_diagnosis.json').write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Savedimmutablecapture/source diagnosis; current live selection admission separately gated in launchplan prep underSH. No retry orGPUrun here.','prior_triage':info(tri),'records':records,'source_evidence':sources,'new_GPU_runs':0,'primary_results_changed':False},indent=2)+'\n')
print('9saveddeadlinecases:dispatch4 progressed executionchecks;spsort4 corrected to2fullpassingtrials;SLUKA4passingtrials+ASICwarmup; no launches')
