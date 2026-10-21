from pathlib import Path
import json,gzip,hashlib,fcntl,collections,datetime
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';bind={};ledgers={};rows=[]
def sha(b):return hashlib.sha256(b).hexdigest()
with open('/tmp/hecbench-roofline-metadata.lock','r')as f:
 fcntl.flock(f,fcntl.LOCK_SH)
 for p in sorted((S/'benchmarks').glob('*.json')):
  b=p.read_bytes();bind[str(p)]=sha(b);v=json.loads(b);ledgers[v['benchmark']]=v
  for m,s in v['models'].items():
   latest=s.get('latest_profile_attempt')or{};selected=s.get('profiling_status');ls=latest.get('status');
   if selected!='profile_process_completed'or(ls and ls!='profile_process_completed'):
    rows.append({'benchmark':v['benchmark'],'model':m,'selected_status':selected,'latest_status':ls,'best_selected_profiles':s.get('profile_ids',[]),'latest_attempt':latest,'plan_status':s.get('profile_plan_status'),'plan_blocks':s.get('profile_plan_blocking_gates',[]),'selected_reason':s.get('reason'),'kernel_observations':s.get('kernel_observations'),'accepted_contract':v.get('accepted_named_contract')})
 for p in [S/'PROGRESS.json',S/'completion_reviews/completion_supervisor/root_monitoring_frontier.json',S/'completion_reviews/completion_supervisor/controller_reviews_v17.json']:
  b=p.read_bytes();bind[str(p)]=sha(b);(D/p.name).write_bytes(b)
 for p,h in bind.items():
  if sha(Path(p).read_bytes())!=h:raise RuntimeError('Drift '+p)
 fcntl.flock(f,fcntl.LOCK_UN)
(D/'ledger_snapshot.json.gz').write_bytes(gzip.compress((json.dumps(ledgers,indent=2)+'\n').encode(),mtime=0));(D/'frontier_rows.json').write_text(json.dumps(rows,indent=2)+'\n');(D/'snapshot_bindings.json').write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_lock':'SH read + final stability rehash','bindings':bind},indent=2)+'\n')
print(json.dumps({'names':len(ledgers),'rows_needing_review':len(rows),'selected_status_counts':dict(collections.Counter(s['profiling_status']for v in ledgers.values()for s in v['models'].values())),'latest_status_counts':dict(collections.Counter((s.get('latest_profile_attempt')or{}).get('status','none')for v in ledgers.values()for s in v['models'].values()))}))
for x in rows:
 if x['selected_status']not in ['profile_process_completed','roofline_not_applicable_no_device_kernels','native_hardware_blocked','hardware_blocked']:print(x['benchmark'],x['model'],x['selected_status'],x['latest_status'])
