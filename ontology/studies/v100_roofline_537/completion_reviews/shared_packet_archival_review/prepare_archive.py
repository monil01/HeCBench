"""CPU/private deterministic archival proposal only. Never writes repository/index."""
from pathlib import Path
import json,hashlib,gzip,fcntl,collections
ROOT=Path('/home/imo/HeCBench_julia');BASE=Path('/tmp/roofline-completion-20261021');OUT=Path(__file__).resolve().parent;STUDY=ROOT/'ontology/studies/v100_roofline_537';INDEX=STUDY/'SHARED_EFFORT_INDEX.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def boundaries(v):
 out={}
 def walk(x):
  if isinstance(x,dict):
   if x.get('boundary')in ['begin','end']and x.get('event_id'):out[x['event_id']]=x
   for value in x.values():walk(value)
  elif isinstance(x,list):
   for value in x:walk(value)
 walk(v);return list(out.values())
def pair(d):
 rows=boundaries(d);begin=[x for x in rows if x['boundary']=='begin'];end=[x for x in rows if x['boundary']=='end']
 if len(begin)!=1 or len(end)!=1:raise RuntimeError('Exact one closed boundary pair required')
 b,e=begin[0],end[0]
 if b['agent']!=e['agent']or b['phase']!=e['phase']:raise RuntimeError('Boundary agent/phase mismatch')
 return {'agent':b['agent'],'phase':b['phase'],'begin_event_id':b['event_id'],'end_event_id':e['event_id'],'begin_usage_timestamp':b.get('usage_event_timestamp'),'end_usage_timestamp':e.get('usage_event_timestamp'),'marker_lag_seconds':[b.get('marker_lag_seconds'),e.get('marker_lag_seconds')]}
packets=[('current537_display_traits_consumer','shared_usage_receipt.json'),('current537_display_refresh_passthrough','shared_usage_receipt.json'),('completion_display_trait_passthrough','shared_usage_receipt.json'),('profiler2025_calibration003','usage_receipt.json'),('profiler2025_method_patch/v5',None)]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);index_bytes=INDEX.read_bytes();index=json.loads(index_bytes);old_pairs={};old_events={}
 for entry in index['receipts']:
  p=ROOT/entry['path'];b=p.read_bytes()
  if sha(b)!=entry['sha256']:raise RuntimeError('Existing index receipt SHA mismatch '+str(p))
  for event in boundaries(json.loads(b)):old_events[event['event_id']]=entry['path']
files=[];joins=[];excluded=[];new=[];pairset=set();eventset=set();inputbind={str(INDEX):sha(index_bytes)}
for name,receipt_name in packets:
 source=BASE/name;dest=Path('ontology/studies/v100_roofline_537/completion_reviews')/name
 receipt=source/receipt_name if receipt_name else None
 if receipt:
  b=receipt.read_bytes();p=pair(json.loads(b));key=(p['agent'],p['begin_event_id'],p['end_event_id'])
  if key in pairset or any(p[k]in eventset for k in ['begin_event_id','end_event_id']):raise RuntimeError('New receipt duplicates window')
  pairset.add(key);eventset.update([p['begin_event_id'],p['end_event_id']])
  already=[old_events.get(p[k])for k in ['begin_event_id','end_event_id']]
  if any(already)and not all(already):raise RuntimeError('Partial overlap with existing index receipt')
  if any(already)and len(set(already))!=1:raise RuntimeError('Pair already indexed in incompatible entries')
  entry={'path':str(dest/receipt_name),'sha256':sha(b),'bytes':len(b),'allocation':'shared_nonexclusive_not_per_benchmark_translation'}
  if not any(already):new.append(entry)
  joins.append({'packet':name,'canonical_receipt':entry,'boundary_pair':p,'index_disposition':'already_indexed_no_append'if any(already)else'append_one_receipt','existing_receipt_path':already[0]if any(already)else None})
 for p in sorted(source.rglob('*')):
  if p.is_symlink():raise RuntimeError('Symlink not admitted '+str(p))
  if not p.is_file():continue
  relative=p.relative_to(source)
  if p.name in ['future_controller_receipt.json','future_follower_receipt.json']or(p.name.startswith('future_')and p.suffix in ['.log','.jsonl'])or p.suffix in ['.lock','.tmp','.new']or'__pycache__'in p.parts:
   excluded.append({'path':str(p),'reason':'Mutable active/future runtime output or cache; not archived as closed static evidence'});continue
  raw=p.read_bytes();h=sha(raw);inputbind[str(p)]=h;compress=len(raw)>262144 and p.suffix!='.gz';data=gzip.compress(raw,mtime=0)if compress else raw;target=dest/relative
  if compress:target=target.with_name(target.name+'.gz')
  payload=OUT/'archive_payload'/target;payload.parent.mkdir(parents=True,exist_ok=True)
  if payload.exists():raise RuntimeError('Fresh archival payload required')
  payload.write_bytes(data)
  files.append({'source_path':str(p),'destination':str(target),'payload_relative':str(payload.relative_to(OUT)),'source_sha256':h,'source_bytes':len(raw),'archived_sha256':sha(data),'archived_bytes':len(data),'compression':'gzip_mtime0'if compress else'none','static_evidence_only':True})
# v5 is already covered by calibration003's one receipt: link, do not synthesize another cost.
cal=next(x for x in joins if x['packet']=='profiler2025_calibration003');calreceipt=json.loads((BASE/'profiler2025_calibration003/usage_receipt.json').read_text());delivery=BASE/'profiler2025_method_patch/v5/DELIVERY.json'
if calreceipt.get('frozenv5_DELIVERY_sha256')!=sha(delivery.read_bytes()):raise RuntimeError('v5 link does not match recorded combined receipt')
joins.append({'packet':'profiler2025_method_patch/v5','canonical_receipt':cal['canonical_receipt'],'boundary_pair':cal['boundary_pair'],'index_disposition':'linked_static_evidence_no_additional_receipt_or_tokens','v5_DELIVERY_sha256':sha(delivery.read_bytes()),'reason':'Combined003 receipt explicitly includes actual002 unit review,003 controller/manifest and privatev5; no cold per-subtask attribution.'})
proposed=dict(index);proposed['receipts']=sorted(index['receipts']+new,key=lambda r:r['path'])
if len({r['path']for r in proposed['receipts']})!=len(proposed['receipts']):raise RuntimeError('Duplicate index path')
# Exact root-window and no-sum policy remain byte-equivalent JSON values.
proposed['updated_at']=max([index['updated_at']]+[x.get('recorded_at','')for j in joins[:4]for x in boundaries(json.loads((BASE/j['packet']/('usage_receipt.json'if j['packet']=='profiler2025_calibration003'else'shared_usage_receipt.json')).read_text()))])
dump(OUT/'proposed_SHARED_EFFORT_INDEX.json',proposed);(OUT/'base_SHARED_EFFORT_INDEX.json').write_bytes(index_bytes)
manifest={'schema_version':1,'status':'PRIVATE_PROPOSAL_ROOT_REVIEW_REQUIRED_NOT_APPLIED','base_index':{'path':str(INDEX.relative_to(ROOT)),'sha256':sha(index_bytes)},'proposed_index':{'payload_relative':'proposed_SHARED_EFFORT_INDEX.json','sha256':sha((OUT/'proposed_SHARED_EFFORT_INDEX.json').read_bytes())},'files':files,'receipt_joins':joins,'new_index_receipts':new,'excluded_runtime_outputs':excluded,'source_bindings':inputbind,'archive_policy':'Preserve static bytes; gzip onlylarge files deterministicmtime0; raw/archivedSHA+length explicit; no candidate adoption/launch; preserve canonical root_window and sum_prohibited_reason.','v5_additional_receipt_count':0,'token_sum':None,'allocation':'shared_nonexclusive_not_per_benchmark_translation','GPU_build_tests_launches_shared_mutations':0}
for p,h in inputbind.items():
 if p==str(INDEX):continue
 if sha(Path(p).read_bytes())!=h:raise RuntimeError('Static packet changed during proposal '+p)
dump(OUT/'archive_manifest.json',manifest)
print(json.dumps({'files':len(files),'raw_bytes':sum(x['source_bytes']for x in files),'archive_bytes':sum(x['archived_bytes']for x in files),'new_index_receipts':len(new),'linked_no_extra_receipt':1,'excluded_runtime_outputs':len(excluded),'manifest_sha256':sha((OUT/'archive_manifest.json').read_bytes())}))
