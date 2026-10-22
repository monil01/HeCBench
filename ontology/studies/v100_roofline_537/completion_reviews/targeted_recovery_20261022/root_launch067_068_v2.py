from pathlib import Path
import json,hashlib,subprocess,datetime,os,shutil
P=Path('/tmp/roofline-completion-20261021');D=P/'testsnap_simplemultidevice_recovery';V=P/'testsnap_simplemultidevice_recovery_storage_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_packet(folder,expected):
 assert sha(folder/'DELIVERY.json')==expected
 doc=json.loads((folder/'DELIVERY.json').read_text())
 for row in doc['files']:
  p=Path(row['path']);assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],str(p)
verify_packet(D,'2119c4b91ffe2cd08988970b059e696b27192496c5e16a2b4a01d5a9a29d0734');verify_packet(V,'93dfaf9b1a1e73d99c2e1ed2544cb00893488181ae4c6417060120332d4954c3')
d=json.loads((D/'plans.json').read_text());assert len(d['plans'])==7
for x in d['source_patches']:
 assert sha(x['source'])==x['source_sha256'];assert sha(x['copy'])==x['copy_sha256'];copy=Path(x['copy']).read_text();assert copy==x['text']
 if x['whole_guard_replacement']:assert Path(x['source']).read_text()==x['original_text']
 else:
  for sub in reversed(x['literal_substitutions']):assert copy.count(sub['after'])>=1;copy=copy.replace(sub['after'],sub['before'])
  assert hashlib.sha256(copy.encode()).hexdigest()==x['source_sha256']
for row in d['plans']:
 p=Path(row['plan']['path']);assert sha(p)==row['plan']['sha256'];plan=json.loads(p.read_text());assert plan['model']in ['cuda','cuda_jl','ka','jacc'];assert plan['reference_result_sha256']==sha(plan['reference_result'])
 for path,h in plan['execution_bindings'].items():assert sha(path)==h,path
commands=json.loads((V/'future_commands.json').read_text());assert len(commands['jobs'])==2
review=P/'root067_068_approval.json';review.write_text(json.dumps({'approved':True,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_packet':sha(D/'DELIVERY.json'),'storage_packet':sha(V/'DELIVERY.json'),'checked_source_overlays':16,'source_math_checks_timer_changes':False,'host_guard_scope':'exact UUID and runtime count outside timing; selected argv1 only','fresh_scratch_policy':'Full SHA/bytes manifest then canonical C21 cleanup of disposable new scratch only; binaryvalues not retained for recertification. Preserve original primary/caches/failed histories.','finite_invocations':2,'GPU':'6only'},indent=2)+'\n')
receipt=P/'root067_068_execution_receipt.json';assert not receipt.exists();state={'status':'active','steps':[],'root_review_sha256':sha(review),'maximum_collector_invocations':2};receipt.write_text(json.dumps(state,indent=2)+'\n')
for j in commands['jobs']:
 gate=subprocess.run(j['prelaunch_space_gate'],capture_output=True,text=True);state['steps'].append({'benchmark':j['benchmark'],'space_gate':gate.stdout,'gate_returncode':gate.returncode});receipt.write_text(json.dumps(state,indent=2)+'\n');assert gate.returncode==0
 argv=j['argv'];assert '--retain-auxiliary'not in argv;assert sha('/home/imo/HeCBench_julia/tools/v100_roofline_collect.py')=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116'
 log=P/('root067_068_'+j['benchmark']+'_collector.log')
 with log.open('xb')as out:done=subprocess.run(argv,cwd='/home/imo/HeCBench_julia',stdout=out,stderr=subprocess.STDOUT)
 state['steps'][-1].update(collector_returncode=done.returncode,log=str(log));receipt.write_text(json.dumps(state,indent=2)+'\n');assert done.returncode==0
 for p in Path(argv[argv.index('--plan-root')+1]).glob('**/profile_provenance.json'):
  prov=json.loads(p.read_text());assert (prov.get('profile')or{}).get('stop_reason')!='external_gpu_activity_detected_own_profile_stopped','New foreign activity; halt remaining finite jobs'
state.update(status='finite_calls_finished_root_scientific_review_required',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());receipt.write_text(json.dumps(state,indent=2)+'\n')
