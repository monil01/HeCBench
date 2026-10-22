from pathlib import Path
import json,hashlib,difflib,fcntl,datetime
B=Path('/tmp/roofline-completion-20261021/su3_tmpfs063_native_v2');V=Path('/tmp/roofline-completion-20261021/su3_tmpfs063_plan_proposal');O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');bind={};future={};h=lambda b:hashlib.sha256(b).hexdigest()
def sha(p):
 p=Path(p);assert not str(p).startswith('/dev/shm');s=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):s.update(b)
 bind[str(p)]={'sha256':s.hexdigest(),'bytes':p.stat().st_size};return s.hexdigest()
def load(p):sha(p);return json.loads(Path(p).read_bytes())
def sub(v,x,y):
 if isinstance(v,str):return v.replace(x,y)
 if isinstance(v,list):return [sub(z,x,y)for z in v]
 if isinstance(v,dict):
  out={}
  for k,z in v.items():nk=sub(k,x,y);assert nk not in out;out[nk]=sub(z,x,y)
  return out
 return v
def dif(a,b,p=()):
 if type(a)!=type(b):return [list(p)]
 if isinstance(a,dict):return [list(p)]if a.keys()!=b.keys()else[d for k in a for d in dif(a[k],b[k],p+(k,))]
 if isinstance(a,list):return [list(p)]if len(a)!=len(b)else[d for i,(x,y)in enumerate(zip(a,b))for d in dif(x,y,p+(i,))]
 return []if a==b else[list(p)]
def readonly(path,expected):
 if path.startswith('/dev/shm'):
  assert path==overlay['fresh_source_path']and expected==h(fresh);future[path]={'sha256':expected,'private_source':overlay['fresh_source_archive'],'not_accessed':True};return
 p=Path(path)
 if path in [str(W/'tools/v100_tmpfs_scratch.py'),str(W/'tools/v100_roofline_collect_tmpfs.py')] and not p.is_file():
  assert la['future_tool_bindings'][path]==expected
  private=B/'future_tools'/p.name;assert sha(private)==expected;future[path]={'sha256':expected,'private_source':str(private),'deployment_required':True};return
 assert sha(p)==expected
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);d=load(B/'DELIVERY.json');dh=sha(B/'DELIVERY.json');assert dh.startswith('85f7d707')and dh.endswith('e6c6b')
 for n,e in d['files'].items():assert sha(B/n)==e['sha256']and (B/n).stat().st_size==e['bytes']
 r=load(B/'root_review_DRAFT_NOT_APPROVED.json');la=load(B/'launch_review_DRAFT_NOT_APPROVED.json');proofs=load(B/'all_four_plan_join_proofs.json');assert len(proofs)==4
 assert r['root_reviewed']is False and r['execution_permission']is False and la['root_reviewed']is False and la['root_authorizes_fresh_tmpfs_staging']is False
 overlay=r['models']['su3/cuda']['native_failure_diagnostic_overlay'];assert overlay['explicit_root_approval']is False;oldbytes=Path(overlay['original_source_archive']).read_bytes();fresh=Path(overlay['fresh_source_archive']).read_bytes();assert (len(oldbytes),len(fresh))==(4503,4507);assert h(oldbytes)==overlay['original_source_sha256']and h(fresh)==overlay['fresh_source_sha256'];x=(overlay['old_scratch_prefix']+'/profile_outputs/private_capture/native_warm_').encode();y=(overlay['new_scratch_prefix']+'/profile_outputs/private_capture/native_warm_').encode();assert oldbytes.count(x)==fresh.count(y)==oldbytes.count(overlay['old_scratch_prefix'].encode())==fresh.count(overlay['new_scratch_prefix'].encode())==1;assert oldbytes.replace(x,y,1)==fresh and fresh.replace(y,x,1)==oldbytes;assert b'if(!ok){std::string path="'+y in fresh;assert sha(overlay['original_source_path'])==h(oldbytes)
 assert (B/'future_tools/v100_roofline_collect_tmpfs.py').read_bytes()==(V/'future_tools/v100_roofline_collect_tmpfs.py').read_bytes()
 for name,old,new in [('helper_minimal.patch',V/'future_tools/v100_tmpfs_scratch.py',B/'future_tools/v100_tmpfs_scratch.py'),('follower_minimal.patch',V/'wait_then_collect.py',B/'wait_then_collect.py')]:
  oldtext=old.read_text();newtext=new.read_text();header='v100_tmpfs_scratch.py'if name.startswith('helper')else'wait_then_collect.py';assert ''.join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile='v1/'+header,tofile='v2/'+header))==(B/name).read_text();sha(old)
 rows=[]
 for m in ['cuda','cuda_jl','ka','jacc']:
  row=r['models']['su3/'+m];p=B/'collector_plans/su3'/m/'plan.json';a=load(p);old=load(row['original_plan_path']);assert sha(p)==row['fresh_plan_sha256']==la['storage_review_required_plan_sha256']['su3/'+m]and sha(row['original_plan_path'])==row['original_plan_sha256'];inv=sub(a,a['scratch_cwd'],old['scratch_cwd']);delta=dif(inv,old)
  if m=='cuda':
   assert delta==[['execution_bindings',overlay['original_source_path']]];assert inv['execution_bindings'][overlay['original_source_path']]==h(fresh)and old['execution_bindings'][overlay['original_source_path']]==h(oldbytes);inv['execution_bindings'][overlay['original_source_path']]=h(oldbytes)
   # Retained proof text follows path substitution; its copy/source SHA declarations remain historical.
   hist=a['remaining_completion_proof'];assert hist['source_bindings'][overlay['original_source_path']]==h(oldbytes)if overlay['original_source_path']in hist['source_bindings']else True
   assert any(q.get('text')==fresh.decode()for q in hist['patches'])
  else:assert delta==[]and p.read_bytes()==(V/'collector_plans/su3'/m/'plan.json').read_bytes()
  assert inv==old
  for pk,rk in [('configuration_id','configuration_id'),('reference_result_sha256','reference_result_sha256'),('raw_config_sha256','raw_config_sha256')]:assert a[pk]==row[rk]
  req={**a['execution_bindings'],a['reference_result']:a['reference_result_sha256'],row['raw_config_path']:a['raw_config_sha256']};assert all(row['read_only_bindings'].get(p)==s for p,s in req.items());assert a.get('runtime_asset_bytes',0)==0
  for pth,s in row['read_only_bindings'].items():readonly(pth,s)
  result=load(a['reference_result']);t=result['models'][m]['trials'];pf=next(q for q in proofs if q['model']==m);tt=pf['five_accepted_primary_trial_metadata'];assert len(t)==len(tt)==5
  for i,(got,meta)in enumerate(zip(t,tt),1):
   for k in ['trial','correct','jit_excluded','time_seconds','internal_repeats','timing_region']:assert got[k]==meta[k]
   assert got['trial']==i and got['correct']is True and got['jit_excluded']is True and got['internal_repeats']==3;assert got['output_artifact']==meta['artifact'];assert meta['artifact']['dtype']=='uint32'and meta['artifact']['count']==336962416;out=pf['fresh_five_expected_outputs'][i-1];assert out=={'path':a['scratch_cwd']+('/su3_trial'if m=='cuda'else'/profile_outputs/su3_trial')+str(i)+'.bin','bytes':1347849664}
  rows.append({'model':m,'old_plan_sha256':row['original_plan_sha256'],'fresh_plan_sha256':row['fresh_plan_sha256'],'inverse_difference_paths_before_native_SHA_map':delta,'entire_original_JSON_recovers':True,'accepted_five_trial_metadata_and_success_bytes_unchanged':True,'five_output_bytes':6739248320})
 assert r['limits']==load(V/'root_review_DRAFT_NOT_APPROVED.json')['limits'];assert r['limits']['attempt_logical_cap_bytes']==12*1024**3 and r['limits']['owned_root_logical_cap_bytes']==32*1024**3
 for pth,s in la['packet_bindings'].items():readonly(pth,s)
 for pth,s in la['legacy_method_bindings'].items():readonly(pth,s)
 for pth,e in list(bind.items()):assert sha(pth)==e['sha256']
 report={'status':'independent_v2_source_bytes_and_whole_plan_inverse_consistent_pending_new_root_approval','reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'delivery_sha256':dh,'collector_sha256':d['collector_unchanged_sha256'],'helper_sha256':d['helper_v2_sha256'],'follower_sha256':d['follower_v2_sha256'],'native_full_bytes_forward_inverse':True,'native_sole_edit_is_host_failure_path_literal':True,'model_plan_reviews':rows,'limits':r['limits'],'runtime_fixture_copy_bytes':0,'native_source_copy_bytes':4507,'all4_five_success_payload_bytes':26956993280,'largest_failure_diagnostic_bytes':335544576,'unchanged_auxiliary_cap_bytes':268435456,'partial_failure_diagnostic_allowed_not_pass':True,'sampled_watchdog_not_hard_quota':True,'future_bindings_proved_from_private_archives_not_runtime_files':future,'no_proposed_code_execution_import_tests_build_GPU_devshm_access':True,'input_bindings':bind}
 (O/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['input_bindings','future_bindings_proved_from_private_archives_not_runtime_files']},indent=2))
