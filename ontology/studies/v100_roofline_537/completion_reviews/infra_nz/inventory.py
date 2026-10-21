import pathlib,json,hashlib,collections
R=pathlib.Path('/tmp/hecbench-roofline-v100-20261020'); W=pathlib.Path('/home/imo/HeCBench_julia'); O=pathlib.Path('/tmp/roofline-completion-20261021/infra_nz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
latest={}
for p in R.glob('*/*/*/profile_provenance.json'):
 try:a=json.loads(p.read_text())
 except:continue
 n,m=p.parent.parent.name,p.parent.name
 if not 'n'<=n[0].lower()<='z':continue
 key=(n,m); order=a.get('started_at','')
 if key not in latest or order>latest[key][0]:latest[key]=(order,p,a)
rows=[]
for p in sorted((W/'ontology/studies/v100_roofline_537/benchmarks').glob('*.json')):
 a=json.loads(p.read_text()); n=a['benchmark']
 if not 'n'<=n[0].lower()<='z':continue
 for m,v in a.get('models',{}).items():
  r={'benchmark':n,'model':m,'ledger':str(p),'ledger_sha256':sha(p),'selected_contract':a.get('accepted_named_contract'), 'selected_status':v.get('profiling_status'),'selected_profile_ids':v.get('profile_ids'),'ledger_latest_attempt':v.get('latest_profile_attempt'),'blocking_gates':v.get('profile_plan_blocking_gates')}
  t=latest.get((n,m))
  if t:
   _,q,b=t;r.update(latest_raw_provenance=str(q),latest_raw_sha256=sha(q),latest_status=b.get('status'),latest_started_at=b.get('started_at'),returncode=b.get('returncode'),reason=b.get('reason'))
   r['files']={str(x):sha(x) for x in q.parent.iterdir() if x.is_file() and x.name in ('plan.json','execution.log','build.log','derived.json','counters.csv')}
  else:r['latest_status']=None
  if r['latest_status'] and r['latest_status']!='profile_process_completed' or r['selected_status'] in ('not_started_plan_blocked','native_hardware_blocked'):
   rows.append(r)
(O/'inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 print(r['benchmark'],r['model'],r['latest_status'] or r['selected_status'],r.get('returncode'),r['blocking_gates'])
 if r.get('latest_raw_provenance'):
  d=pathlib.Path(r['latest_raw_provenance']).parent
  for name in ['build.log','execution.log']:
   f=d/name
   if f.exists():print(name, f.read_text(errors='replace')[-2300:].replace('\n',' | '))
