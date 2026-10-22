# Independent CPU source/data inspection; does not import candidate scripts.
import json,gzip,hashlib,fcntl,re,collections
from pathlib import Path
D=Path('/tmp/roofline-completion-20261021/roi_membership_source_batch10_ln');OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda o:sha(json.dumps(o,sort_keys=True,separators=(',',':')).encode())
def load(raw,path):return json.loads(gzip.decompress(raw)if str(path).endswith('.gz')else raw)
raw=(D/'proposed_stage_patches.json').read_bytes();doc=json.loads(raw);issues=[];stats=collections.Counter();cache={}
def issue(kind,where,detail):issues.append({'kind':kind,'where':where,'detail':str(detail)})
paths=set(doc['parent_reviews'])
for r in doc['selected_capture_bindings']:
 for k in ('ledger','manifest','plan','derivation','provenance'):paths.add(r[k]['path'])
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);snapshot={p:Path(p).read_bytes()for p in sorted(paths)}
parents={p:load(snapshot[p],p)for p in doc['parent_reviews']}
for p,b in doc['parent_reviews'].items():
 if sha(snapshot[p])!=b['compressed_sha256']or sha(gzip.decompress(snapshot[p]))!=b['uncompressed_sha256']:issue('parent_drift',p,'compressed/raw SHA mismatch')
def filebytes(path):
 if path not in cache:cache[path]=Path(path).read_bytes()
 return cache[path]
def verify_file(path,digest,where):
 try:
  if sha(filebytes(path))!=digest:issue('file_SHA_mismatch',where,path)
 except OSError as e:issue('file_missing',where,e)
def verify_quote(item,where):
 p=item['path'];digest=item.get('sha256',item.get('source_sha256'));verify_file(p,digest,where)
 try:text=filebytes(p).decode('utf8')
 except UnicodeDecodeError:text=filebytes(p).decode('latin1')
 lines=text.splitlines()
 for line in item.get('line_evidence',[]):
  if lines[line['line']-1]!=line['text']:issue('line_quote_mismatch',where,(p,line['line']))
  stats['exact_line_quotes']+=1
 if 'context'in item:
  start=item['context_start_line']-1;quote=item['context'];actual='\n'.join(lines[start:start+len(quote.splitlines())])
  if actual!=quote.rstrip('\n'):issue('callsite_context_mismatch',where,p)
  stats['exact_context_quotes']+=1
for source in doc['sources']:verify_file(source['path'],source['sha256'],'sources');stats['source_files']+=1
selected={(r['benchmark'],r['model']):r for r in doc['selected_capture_bindings']}
for p in doc['patches']:
 key=p['record_key'];where=key['benchmark']+'/'+key['model'];parent=parents[p['parent_review']['path']]['records'][p['record_index']]
 if canon(parent)!=p['expected_original_record_canonical_sha256']:issue('record_hash_mismatch',where,'parent')
 if any(parent[k]!=key[k]for k in ('benchmark','model','configuration_id')):issue('record_identity',where,key)
 for k,v in p['unchanged_record_qualifications'].items():
  if parent[k]!=v:issue('unchanged_qualification_mismatch',where,k)
 r=selected[(key['benchmark'],key['model'])];objects={}
 for label in ('ledger','manifest','plan','derivation','provenance'):
  b=snapshot[r[label]['path']]
  if sha(b)!=r[label]['sha256']:issue('selected_file_drift',where,label)
  objects[label]=load(b,r[label]['path'])
 plan=objects['plan'];prov=objects['provenance'];der=objects['derivation'];ledger=objects['ledger']['models'][key['model']]
 root=Path('/home/imo/HeCBench_julia')
 if str((root/ledger['profile_evidence_manifest']).resolve())!=str(Path(r['manifest']['path']).resolve()):issue('selected_manifest_pointer',where,ledger['profile_evidence_manifest'])
 for label,original_name in [('plan','plan.json'),('provenance','profile_provenance.json'),('derivation','derived.json')]:
  bound=next(x for x in objects['manifest']['files']if x['original_name']==original_name)
  rawbytes=snapshot[r[label]['path']]
  if bound['compressed_sha256']!=sha(rawbytes)or bound['uncompressed_sha256']!=sha(gzip.decompress(rawbytes)):issue('manifest_compressed_raw_join',where,label)
 if prov['plan_sha256']!=sha(gzip.decompress(snapshot[r['plan']['path']])):issue('provenance_plan_raw_join',where,'SHA')
 counters=next(x for x in objects['manifest']['files']if x['original_name']=='counters.csv')
 if der['input_sha256']!=counters['uncompressed_sha256']or prov['counters_sha256']!=counters['uncompressed_sha256']:issue('derived_counter_manifest_join',where,'SHA')
 if plan['configuration_id']!=key['configuration_id']:issue('CID_mismatch',where,'selected plan')
 contract=objects['ledger'].get('accepted_named_contract',{})
 if contract.get('configuration_id')!=key['configuration_id']:issue('CID_mismatch',where,contract)
 if prov.get('status')!='profile_process_completed':issue('not_selected_success',where,prov.get('status'))
 if contract.get('result_sha256')!=plan['reference_result_sha256'] or contract.get('raw_config_sha256')!=plan['raw_config_sha256']:issue('canonical_contract_join',where,contract)
 captured={**plan.get('execution_bindings',{}),**prov.get('additional_current_profile_input_bindings',{})}
 captured_resolved={str(Path(path).resolve()):digest for path,digest in captured.items()}
 included={};patched={}
 for s in p['stage_patches']:
  old=parent['stages'][s['stage_index']];u=s['updates'];q=u['ROI_membership_proof'];stage=s['stage_id']
  if old['stage_id']!=stage or canon(old)!=s['expected_original_stage_canonical_sha256']:issue('stage_hash_identity',where,stage)
  if q.get('sample_in_primary_timed_invocation')is not None or q.get('retained_trial')is not None or q.get('whole_elapsed_ROI_claim')is not False or q.get('complete_ROI')is not False or q.get('causal_bottleneck')!='unknown':issue('scope_promotion',where,stage)
  for field in ('selected_result','selected_config'):
   bound=q[field];verify_file(bound['path'],bound['sha256'],where)
  if q['selected_result']['sha256']!=plan['reference_result_sha256']or q['selected_config']['sha256']!=plan['raw_config_sha256']:issue('result_config_join',where,stage)
  config=json.loads(filebytes(q['selected_config']['path']))
  if config.get('configuration_id')!=key['configuration_id']:issue('frozen_config_CID_join',where,config.get('configuration_id'))
  site=q['exact_stage_callsite']
  if captured.get(site['path'],captured_resolved.get(str(Path(site['path']).resolve())))!=site['source_sha256']:
   issue('callsite_source_not_captured',where,site['path'])
  verify_quote(site,where+'/'+stage)
  for field in ('evidence','timed_closure_boundaries'):
   for item in q[field]:verify_quote(item,where+'/'+stage)
  updated=dict(old);updated.update(u);patched[stage]=updated
  if u['semantic_family_in_primary_ROI']is True:included[stage]=updated;stats['inclusions']+=1
  else:stats['exclusions']+=1
  stats['stage_updates']+=1
 matches=[]
 for row in der['records']:
  # Actual parser observations carry kernel identity under identity.
  name=row.get('identity',{}).get('Kernel Name')
  if name is None:name=row.get('kernel',{}).get('Kernel Name')
  if name is None:issue('unknown_derived_identity_schema',where,list(row));continue
  ids=[sid for sid,s in included.items()if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],name)]
  if len(ids)>1:issue('ambiguous_family',where,ids)
  if ids:matches.extend(ids)
 if not matches:issue('no_positive_included_family',where,'no full selected saved family match')
 stats['model_records']+=1
# Overlay actual retained plan binds + literal/path-only transforms independently repeated.
overlay_path=D/'linearprobing_retained_overlay_proof.json.gz';overlayraw=overlay_path.read_bytes();overlay=json.loads(gzip.decompress(overlayraw));overlaychecks=[]
for r in overlay['reviewed_models']:
 plan=load(Path(r['selected_plan']['path']).read_bytes(),r['selected_plan']['path']);check={'model':r['model'],'source_copies':[]}
 if sha(Path(r['selected_plan']['path']).read_bytes())!=r['selected_plan']['sha256']:issue('overlay_plan_SHA',r['model'],'drift')
 patches=plan['capture_overlay_provenance']['patches']
 for o in r['overlays']:
  original=filebytes(o['original']);text=o['retained_text'].encode();record=next((x for x in patches if x['copy']==o['copy']),None)
  if record is None:issue('overlay_missing_capture_patch',r['model'],o['copy']);continue
  if sha(original)!=o['original_sha256']or sha(text)!=o['copy_sha256']or record['text'].encode()!=text or record['copy_sha256']!=o['copy_sha256']or plan['execution_bindings'].get(o['copy'])!=o['copy_sha256']:issue('overlay_binding',r['model'],o['copy'])
  current=original.decode();ok=True
  for t in o['reviewed_exact_changes']:
   if current.count(t['before'])!=1:issue('overlay_nonunique_transform',r['model'],t);ok=False;break
   current=current.replace(t['before'],t['after'],1)
  if current.encode()!=text:issue('overlay_forward_bytes',r['model'],o['copy'])
  for t in reversed(o['reviewed_exact_changes']):
   if current.count(t['after'])!=1:issue('overlay_nonunique_inverse',r['model'],t);ok=False;break
   current=current.replace(t['after'],t['before'],1)
  if current.encode()!=original:issue('overlay_inverse_bytes',r['model'],o['copy'])
  if Path(o['copy']).exists()and sha(Path(o['copy']).read_bytes())!=o['copy_sha256']:issue('present_copy_drift',r['model'],o['copy'])
  check['source_copies'].append({'original':o['original'],'copy':o['copy'],'actual_retained_SHA':sha(text),'original_SHA':sha(original),'forward_inverse_exact':ok and current.encode()==original})
 overlaychecks.append(check)
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);end={p:sha(Path(p).read_bytes())for p in sorted(snapshot)}
changed=[p for p,b in snapshot.items()if end[p]!=sha(b)]
if changed:issue('end_snapshot_drift','metadata',changed)
source_end_changed=[s['path']for s in doc['sources']if sha(Path(s['path']).read_bytes())!=s['sha256']]
if source_end_changed:issue('end_source_drift','source',source_end_changed)
result={'source_end_stable':not source_end_changed,'proposal_sha256':sha(raw),'overlay_compressed_sha256':sha(overlayraw),'stats':dict(stats),'issues':issues,'issue_count':len(issues),'overlay_checks':overlaychecks,
 'metadata_snapshot_count':len(snapshot),'metadata_end_stable':not changed,'snapshot_scope':'Short shared-lock beginning/end snapshots; source read/analysis outside lock. Not simultaneous GPU evidence.',
 'no_candidate_import_or_execution':True,'no_GPU_build_tests_or_shared_mutations':True}
(OUT/'static_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stats':dict(stats),'issue_count':len(issues),'first_issues':issues[:8],'overlay_SHA':sha(overlayraw)}))
