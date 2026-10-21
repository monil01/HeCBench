from pathlib import Path
import json,hashlib,fcntl,csv,io,datetime,gzip
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def meta(p):b=Path(p).read_bytes();return {'path':str(p),'sha256':sha(b),'bytes':len(b)}
snapshot=json.loads((O/'snapshot.json').read_text());actions=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 rp=S/'completion_reviews/infra_am/runtime_no_kernel_applicability.json';review=json.loads(rp.read_text());rb=meta(rp);known={(r['benchmark'],r['model']):r for r in review['rows']}
 for r in snapshot['rows']:
  key=r['benchmark'],r['model'];v=r['provenance'];plan=r['plan'];rv=known[key];lp=Path(r['ledger']['path']);live=json.loads(lp.read_text())['models'][r['model']]
  rawprovsha=r['files']['profile_provenance.json']['uncompressed_sha256'];assert rv['parent_profile']['sha256']==rawprovsha
  assert live['profiling_status']=='roofline_not_applicable_no_device_kernels'
  assert live['profile_applicability_review']['sha256']==rb['sha256']
  assert (v['profile']['returncode'],v['profile']['stop_reason'])==(0,None)
  lines=r['counter_csv'].splitlines();numeric_header=any(line.startswith('"ID",')or line.startswith('ID,')for line in lines)
  assert not numeric_header and 'No kernels were profiled.'in r['counter_csv']and v['counter_parse_error']=='No NCU CSV ID header'
  trials=[json.loads(line[len('STUDY_TRIAL '):])for line in r['execution_log'].splitlines()if line.startswith('STUDY_TRIAL ')]
  assert [t['trial']for t in trials]==[1,2,3,4,5]and all(t['correct']is True for t in trials)
  source=[]
  for e in rv['source_review']:
   p=Path(e['path']);actual=meta(p)
   assert actual['sha256']==e['sha256']
   text=p.read_text();matches=[{'line':i+1,'text':line}for i,line in enumerate(text.splitlines())if any(x in line for x in ['cudaMalloc','cudaFree','cudaHostAlloc','cudaMallocHost','cudaFreeHost','dlsym','transfer!','copyto!','unsafe_copyto!','cuMemcpy','CUDA.Mem','ccall','study_batch'])]
   source.append({**actual,'source_declared_API_lines':matches,'captured_parent_source_binding':e.get('parent_supplemental_sha256'),'historical_qualification':'Existing hash-bound applicability source review plus captured parent supplemental/execution binding; current hash match is supporting static review, not substitute historical execution proof.'})
  rawslot=Path(rv['parent_profile']['path']).parent
  otherplans=[str(p)for p in Path('/tmp/hecbench-roofline-v100-20261020').glob('batch*/'+r['benchmark']+'/'+r['model']+'/plan.json')if p.parent!=rawslot]
  actions.append({'benchmark':r['benchmark'],'model':r['model'],'configuration_id':plan['configuration_id'],'ledger_at_review':meta(lp),'latest_attempt_raw_status':v['status'],'selected_effective_status':live['profiling_status'],'actual_parse_failure':'No NCU CSV ID header:335–461-byte log-only report explicitly says no kernels profiled; no numeric hardware samples exist.','process_outcome':{k:v['profile'][k]for k in ['returncode','stop_reason','started_at','finished_at']},'completed_checked_trials':[{'trial':t['trial'],'correct':t['correct'],'timing_region':t['timing_region'],'no_output_reason':t.get('no_output_reason'),'exact_word_mismatches':t.get('exact_word_mismatches')}for t in trials],'numeric_counter_rows_available':False,'numeric_counter_values':None,'source_applicability_review':{**rb,'entry':r['benchmark']+'/'+r['model'],'exact_parent_provenance_sha256':rawprovsha},'actual_frozen_files':r['files'],'source_review':source,'same_model_other_raw_plans':otherplans,'required_recovery_action':'NONE_FOR_CURRENT_KERNEL_ROOFLINE_CONTRACT_ALREADY_SOURCE_NA','queued_repair_required':False,'NCU2025_capture_required':False,'new_method_if_question_changes':'Allocation latency/runtime API timeline or pageable transfer/PCIe bandwidth needs a separately named measurement protocol and permission; it cannot provide scalarFP kernel points for these unchanged source scopes. Do not launch as a repair of missing counters.','preserve_history':['Retain original raw counter_parse_error and latest_attempt_status unchanged.','Retain current selected classification_only source-NA authority; no synthetic derived.json or zeroFLOPs/zeroDRAM.','No inference that runtime/driver emits no internalSMwork; only no profiled kernel samples and no authored device algorithm in this contract.','No global parser relaxation admitting log-only CSV as valid numeric evidence.'],'source_or_checker_change_proposed':False})
report={'schema_version':1,'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot':meta(O/'snapshot.json'),'cases':len(actions),'count_scope':'Seven historical latest-attempt parser errors; all seven selected effective outcomes already source no-device-kernel NA, not unresolved kernel positioning jobs.','actions':actions,'new_GPU_jobs_proposed':0,'existing_parser_recovery_heartwall_not_applicable':'Heartwall recovery requires actual numeric rows and saved duration/DRAM/PCI checks; these reports contain none. No source/hash-preserving parser can recover unavailable counters.','unrelated_memcpy_native':'Native memcpy current blocked source/path plan is not among these seven and requires its own source-isolation review; no conclusion about that slot.','global_constraints':'Collector32a9/classifier/calibration003 remain untouched and unadopted; no launch/build/test/importtarget/GPUquery/rawmutation/sharedmetadata mutation. Primarytimings unchanged.'}
(O/'reviewed_actions.json').write_text(json.dumps(report,indent=2)+'\n');print('actions',len(actions),'jobs0','sha',sha((O/'reviewed_actions.json').read_bytes()))
