from pathlib import Path
import json,hashlib,collections,datetime,gzip
S=Path('/home/imo/HeCBench_julia');R=Path('/tmp/hecbench-roofline-v100-20261020');D=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for p in sorted((S/'ontology/studies/v100_roofline_537/benchmarks').glob('*.json')):
 if p.stem[0].lower() not in 'abcdefghijklm':continue
 for m,v in json.loads(p.read_text()).get('models',{}).items():
  if v.get('profiling_status')=='profile_process_completed':continue
  row={'benchmark':p.stem,'model':m,'selected_status':v.get('profiling_status'),'ledger':str(p),'ledger_snapshot_sha256':sha(p),'gates':v.get('profile_plan_blocking_gates',[]),'selected_profiles':v.get('profile_ids',[])}
  ids=row['selected_profiles'];slot=R/Path(ids[-1]).name/p.stem/m if ids else None
  if slot and (slot/'profile_provenance.json').exists():
   q=slot/'profile_provenance.json';x=json.loads(q.read_text());pr=x.get('profile',{});row.update(raw_slot=str(slot),provenance_sha256=sha(q),status=x.get('status'),finished_at=x.get('finished_at'),stop_reason=pr.get('stop_reason'),returncode=pr.get('returncode'),wall_seconds=pr.get('wall_seconds'),error=x.get('error'),parse_error=x.get('counter_parse_error'),kernel_count=x.get('kernel_count'))
   for n in ['plan.json','execution.log','counters.csv','build.log']:
    f=slot/n
    if f.exists():row[n]={'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size}
   if (slot/'execution.log').exists():row['log_tail']=(slot/'execution.log').read_text(errors='replace')[-4500:]
  else:row['status']=row['selected_status']
  category='unreviewed'
  if row.get('stop_reason')=='external_gpu_activity_detected_own_profile_stopped' or 'existing compute process' in str(row.get('error')):category='external_contention_fresh_retry'
  elif row.get('stop_reason')=='wall_timeout':category='deadline_or_expensive_replay_review'
  elif row['status']=='native_hardware_blocked':category='documented_native_hardware_barrier'
  elif row['status']=='not_started_replay_method_blocked':category='replay_method_limitation'
  elif row['status']=='not_started_plan_blocked':category='plan_gate_requires_review'
  elif row['status']=='profile_completed_counter_derivation_failed':category='completed_counter_missing_or_parser_review'
  row['category']=category;rows.append(row)
out={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':dict(collections.Counter(x['category']for x in rows)),'rows':rows};(D/'selected_failure_scan.json').write_text(json.dumps(out,indent=2)+'\n');print(out['counts'])
for x in rows:
 if x['category']=='unreviewed':print(x['benchmark'],x['model'],x.get('returncode'),x.get('error'),x.get('log_tail','')[-1100:].replace('\n',' ') )
