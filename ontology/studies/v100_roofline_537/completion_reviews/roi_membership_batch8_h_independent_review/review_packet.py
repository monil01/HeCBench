from pathlib import Path
import json,gzip,hashlib,re,fcntl,datetime
O=Path(__file__).parent;D=O.with_name('roi_membership_source_batch8_h');W=Path('/home/imo/HeCBench_julia');P=D/'proposed_stage_patches.json';sha=lambda b:hashlib.sha256(b).hexdigest();H=lambda p:sha(Path(p).read_bytes());canon=lambda x:sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode());d=json.loads(P.read_bytes());checks=[];issues=[]
def ck(name,yes,detail=None):
 checks.append({'check':name,'passed':bool(yes),'detail':detail})
 if not yes:issues.append({'check':name,'detail':detail})
ck('exact_reviewed_packet',H(P)=='c3433c7e8c2df3712460c69323a4ba833d90df4185069a32480041140eed1cff')
ck('six_Julia_only_records',len(d['patches'])==6 and sorted((r['record_key']['benchmark'],r['record_key']['model'])for r in d['patches'])==sorted((b,m)for b in ['halo-finder','heartwall']for m in ['cuda_jl','ka','jacc']))
ck('native_both_withheld',{(r['benchmark'],r['model'])for r in d['withheld_models']}=={('halo-finder','cuda'),('heartwall','cuda')})
ck('phase_and_fullROI_false_top',d['sample_phase_still_unproved']is True and d['elapsed_ROI_aggregate_claim']is False and d['causal_claim']is False)
parents={};sources={};captures={};matched=0;unassigned=0
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for path,h in d['parent_reviews'].items():
  raw=Path(path).read_bytes();ck('parent_compressed_hash',sha(raw)==h['compressed_sha256'],path);u=gzip.decompress(raw);ck('parent_uncompressed_hash',sha(u)==h['uncompressed_sha256'],path);parents[path]=json.loads(u)
 for row in d['sources']:ck('present_source_hash',H(row['path'])==row['sha256'],row['path']);sources[row['path']]=row['sha256']
 for row in d['selected_capture_bindings']:
  for k in ['ledger','manifest','plan','derivation','provenance']:ck('selected_capture_hash',H(row[k]['path'])==row[k]['sha256'],{'benchmark':row['benchmark'],'model':row['model'],'kind':k})
  captures[(row['benchmark'],row['model'])]=row
 for r in d['patches']:
  key=r['record_key'];old=parents[r['parent_review']['path']]['records'][r['record_index']];tag=key['benchmark']+'/'+key['model'];ck('exact_parent_key_index',all(old[k]==v for k,v in key.items()),tag);ck('canonical_record_hash',canon(old)==r['expected_original_record_canonical_sha256'],tag)
  c=captures[(key['benchmark'],key['model'])];pl=json.loads(gzip.decompress(Path(c['plan']['path']).read_bytes()));ob=json.loads(gzip.decompress(Path(c['derivation']['path']).read_bytes()))['records'];ck('selected_CID_config_result_joins',pl['configuration_id']==old['configuration_id']and pl['raw_config_sha256']==old['raw_config_sha256']and pl['reference_result_sha256']==old['reference_result']['sha256'],tag)
  for k,v in r['unchanged_record_qualifications'].items():ck('unchanged_record_qualifier',old[k]==v,{'slot':tag,'key':k})
  new=[dict(s)for s in old['stages']]
  for sp in r['stage_patches']:
   st=old['stages'][sp['stage_index']];ck('canonical_stage_hash',st['stage_id']==sp['stage_id']and canon(st)==sp['expected_original_stage_canonical_sha256'],tag);u=sp['updates'];ck('source_only_update_fields',set(u)=={'semantic_family_in_primary_ROI','ROI_membership_status','ROI_membership_proof'},tag);p=u['ROI_membership_proof'];ck('source_inclusion_only',u['semantic_family_in_primary_ROI']is True and u['ROI_membership_status']=='source_timed_closure_inclusion_proved',tag)
   ck('proof_no_phase_or_causal_promotion',p['sample_in_primary_timed_invocation']is not True and p['retained_trial']is None and p['whole_elapsed_ROI_claim']is False and p['complete_ROI']is False and p['causal_bottleneck']=='unknown',tag)
   ck('proof_selected_capture_join',p['selected_result']==old['reference_result']and p['selected_config']==old['raw_config']and p['actual_profile_source_bindings']['sha256']==c['plan']['sha256'],tag)
   for ev in p['evidence']:ck('proof_evidence_source_hash',sources.get(ev['path'])==ev['sha256'],{'slot':tag,'path':ev['path']})
   new[sp['stage_index']].update(u)
  actualmatches=[]
  for x in ob:
   hits=[s['stage_id']for s in new if s.get('semantic_family_in_primary_ROI')is True and s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],x['identity']['Kernel Name'])];ck('unique_actual_saved_family_match',len(hits)<=1,tag);actualmatches.append((x['identity'],hits));matched+=bool(hits);unassigned+=not bool(hits)
  ck('model_positive_saved_match',any(v for _,v in actualmatches),tag)
  supplied=[(x['identity'],x['matched_stage_ids'])for x in r['selected_observation_matching']];ck('recomputed_saved_match_equality',actualmatches==supplied,tag)
 fcntl.flock(lock,fcntl.LOCK_UN)
# Independently selected concise scientific source lines, no target imports.
scientific=[]
for rel,terms in [
 ('src/halo-finder-julia/host_driver.cu',['static void observe','excluded+=','auto begin=Clock::now()','seconds+=','source_host_pinned_UVA_mapping_verified','cudaMemoryTypeHost']),
 ('src/halo-finder-julia/configured_native/RCBForceTree.cxx',['halo_observer_callback(0','halo_target_callback(&args)','halo_observer_callback(1','cudaStreamSynchronize(stream)']),
 ('src/halo-finder-julia/study_common.jl',['halo_callback','launch_halo(a,c)','ccall(HALO_RUN','for trial in 0:5']),
 ('src/heartwall-julia/study_host.jl',['for frame=0:FRAMES-1','setframe(frame);append!','start=time_ns();launch','seconds+=','actual_untimed_full_frame_state']),
 ('src/heartwall-julia/heartwall-julia-cuda.jl/kernel.jl',['frame_no == Int32(0)','frame_no != Int32(0)','frame_no % Int32(10)'])]:
 p=W/rel;lines=p.read_text().splitlines();hits=[{'line':i+1,'text':l}for i,l in enumerate(lines)if any(t in l for t in terms)];ck('critical_source_snippets_present',bool(hits),rel);ck('critical_source_in_packet',sources.get(str(p))==H(p),rel);scientific.append({'path':str(p),'sha256':H(p),'lines':hits})
report={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packet':{'path':str(P),'sha256':H(P)},'adoption_admissible_subject_to_root_parent_INDEX_atomic_revalidation':not issues,'blockers':issues,'checks_total':len(checks),'checks_passed':sum(x['passed']for x in checks),'checks':checks,'scientific_source_snippets':scientific,'saved_matching_recomputed':{'matched':matched,'unassigned':unassigned,'models':6},'scientific_adjudication':{'halo':'Explicit source forcecallback enters net tree/workspace/traversal timer; observer0/1includinginputecho are individually subtracted. Devicecompletion before observer1 remains outside observer stopwatch. Mapped pinned/UVA data require nonHBM-only tier; source ForceLaw mismatch qualifier retained, not full CLI gravity fidelity.','heartwall':'Frame0 initializes template, frames1..10track, nonzero frame%10=0updates template. Samehwcallable is in11frame launch+completion timer; uploads/inputcapture beforetimer; fullauxiliary replay afterseries. Firstsignaturematch cannot identify whichbranch/frame/warmup/replay was sampled.','native':'No native stage or inferred scientific kernel introduced. NativeHalohas only excludedobserver candidate; nativeHeartwallunresolved. Holds preserve scientific/profiling/numeric evidence.'},'precision':'SourceFloat32force/correlation/workspace plus integerpointindices; counterclass/SFU/vector/atomics not inferred ceiling/causal proof.','runtime_sample_phase':'unproved','whole_ROI_claim':False,'causal_claim':False,'GPU_runs':0,'build_runs':0,'tests':0,'shared_mutations':False}
q=O/'independent_review.json';q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'adoption_admissible':not issues,'checks':len(checks),'issues':issues,'matches':matched,'unassigned':unassigned,'review_sha256':H(q)}))
