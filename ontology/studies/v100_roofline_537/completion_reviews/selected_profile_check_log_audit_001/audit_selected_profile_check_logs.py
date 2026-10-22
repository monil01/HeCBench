from pathlib import Path
import json,gzip,hashlib,fcntl,collections,datetime
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';O=Path('/tmp/roofline-completion-20261021/selected_profile_check_log_audit_001');O.mkdir(exist_ok=False)
def digest(b):return hashlib.sha256(b).hexdigest()
def member(folder,manifest,name):
 rows=[x for x in manifest['files']if x['original_name']==name]
 if not rows:return None
 assert len(rows)==1;x=rows[0];p=R/x['path'];assert p.parent==folder;packed=p.read_bytes();raw=gzip.decompress(packed);assert digest(packed)==x['compressed_sha256']and digest(raw)==x['uncompressed_sha256']and len(raw)==x['uncompressed_bytes'];return raw
rows=[];counts=collections.Counter()
with Path('/tmp/hecbench-roofline-metadata.lock').open()as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for ledger in sorted((S/'benchmarks').glob('*.json')):
  doc=json.loads(ledger.read_bytes())
  for model,state in doc['models'].items():
   if state['profiling_status']!='profile_process_completed':continue
   ids=state['profile_ids'];assert len(ids)==1;folder=R/ids[0];mp=folder/'manifest.json';mb=mp.read_bytes();manifest=json.loads(mb);pb=member(folder,manifest,'profile_provenance.json');prov=json.loads(pb);log=member(folder,manifest,'execution.log');trials=[];errors=[]
   if log is not None:
    for line in log.splitlines():
     if line.startswith(b'STUDY_TRIAL '):
      try:trials.append(json.loads(line[len(b'STUDY_TRIAL '):]))
      except Exception as e:errors.append(repr(e))
   ids_seen=[x.get('trial')for x in trials];correct=[x.get('correct')for x in trials]
   if errors:category='trial_JSON_parse_error'
   elif prov.get('isolated_assertion_phases'):category='dedicated_assertion_frozen_audit_required'
   elif prov.get('application_replay'):
    groups=[trials[i:i+5]for i in range(0,len(trials),5)];category='all_application_five_check_groups_pass'if groups and all([x.get('trial')for x in g]==list(range(1,6))and all(x.get('correct')is True for x in g)for g in groups)else'application_five_check_group_missing_or_false'
   elif ids_seen==list(range(1,6))and all(x is True for x in correct):category='five_profile_log_checks_pass'
   elif any(x is False for x in correct):category='visible_profile_check_false'
   elif not trials:category='no_standard_trial_markers_source_review_required'
   else:category='trial_cardinality_or_order_or_boolean_unresolved'
   counts[category]+=1;rows.append({'benchmark':doc.get('benchmark',ledger.stem),'model':model,'profile_id':ids[0],'manifest_sha256':digest(mb),'provenance_sha256':digest(pb),'execution_log_sha256':digest(log)if log is not None else None,'category':category,'trial_ids':ids_seen,'correct_flags':correct,'parse_errors':errors,'process_returncode':prov.get('profile',{}).get('returncode'),'process_stop_reason':prov.get('profile',{}).get('stop_reason')})
result={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only selected frozen profile execution-log audit; standard marker observations only, not rechecking binary payload values or changing primary acceptance. Absence requires exact source/dedicated method review; rc0 is not numerical certification.','selected_completed_models':len(rows),'categories':dict(counts),'rows':rows,'GPU_execution':0,'shared_scientific_mutations':0};(O/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
