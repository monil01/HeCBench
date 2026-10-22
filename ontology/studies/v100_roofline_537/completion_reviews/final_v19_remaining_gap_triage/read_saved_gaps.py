# Read-only snapshot/categorization aid, not campaign tool execution.
from pathlib import Path
import json,gzip,hashlib,fcntl,collections
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(p):return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 progress=(S/'PROGRESS.json').read_bytes();ledgers={str(p):p.read_bytes()for p in sorted((S/'benchmarks').glob('*.json'))}
rows=[]
for path,raw in ledgers.items():
 doc=json.loads(raw)
 for model,state in doc['models'].items():
  if state['profiling_status']=='profile_process_completed':continue
  row={'benchmark':doc['benchmark'],'model':model,'status':state['profiling_status'],'ledger':{'path':path,'sha256':sha(raw)},
       'state':state,'accepted_named_contract':doc.get('accepted_named_contract'),
       'selected':state.get('profile_ids',[]),'latest':state.get('latest_profile_attempt',{}).get('profile_id'),'attempts':[]}
  selected=list(state.get('profile_ids',[]));latest=state.get('latest_profile_attempt',{}).get('profile_id')
  for identity in dict.fromkeys(selected+([latest]if latest else [])):
   slot=R/identity
   if not slot.is_dir():row['attempts'].append({'profile_id':identity,'directory_missing':True});continue
   member=load(slot/'manifest.json');a={'profile_id':identity,'manifest_sha256':sha((slot/'manifest.json').read_bytes()),'files':{},'logs':{}}
   for name in ('plan.json','profile_provenance.json'):
    q=slot/(name+'.gz');data=gzip.decompress(q.read_bytes());a['files'][name]={'compressed_sha256':sha(q.read_bytes()),'raw_sha256':sha(data),'value':json.loads(data)}
   for name in ('execution.log','build.log','profiler_version.log','pre_profile_checks.log','post_profile_checks.log'):
    entry=next((e for e in member['files']if e['original_name']==name),None)
    if entry is None:continue
    q=R/entry['path'];packed=q.read_bytes();h=hashlib.sha256();head=b'';tail=b'';size=0
    with gzip.open(q,'rb')as f:
     for block in iter(lambda:f.read(1<<20),b''):
      h.update(block);size+=len(block)
      if len(head)<12000:head=(head+block)[:12000]
      tail=(tail+block)[-18000:]
    a['logs'][name]={'path':str(q),'compressed_sha256':sha(packed),'uncompressed_sha256':h.hexdigest(),'bytes':size,
      'manifest_joins':sha(packed)==entry['compressed_sha256']and h.hexdigest()==entry['uncompressed_sha256']and size==entry['uncompressed_bytes'],
      'head':head.decode(errors='replace'),'tail':tail.decode(errors='replace')}
   row['attempts'].append(a)
  rows.append(row)
(O/'snapshot_all_noncompleted.json').write_text(json.dumps({'progress':json.loads(progress),'progress_sha256':sha(progress),'rows':rows},indent=2)+'\n')
for r in rows:
 if r['status']in('native_hardware_blocked','roofline_not_applicable_no_device_kernels','not_started_plan_blocked'):continue
 print('\nCASE',r['benchmark'],r['model'],r['status'])
 for a in r['attempts']:
  if 'files'not in a:continue
  p=a['files']['profile_provenance.json']['value'];print(a['profile_id'],'status',p.get('status'),'error',p.get('error'),'parse',p.get('counter_parse_error'),'method',p.get('profiler_version'))
  for k in ('profile','build'):print(k,{x:p.get(k,{}).get(x)for x in ('returncode','stop_reason','wall_seconds','external_gpu_activity')})
  for name,log in a['logs'].items():
   if name=='profiler_version.log':continue
   text=log['head']+'\n'+log['tail'];lines=[s for s in text.splitlines()if any(w in s.lower()for w in ('error','failed','assert','exception','traceback','no kernel','not found','undefined','permission','terminate','mpi_abort','no such','illegal','sigsegv','read-only','out of memory','safe_close'))]
   print(name,'diagnostics',lines[-6:])
