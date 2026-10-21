#!/usr/bin/env python3
"""Validate private source-ROI packets; default mode writes private reports only.
--apply is a root-coordinator action: exact preapproved plan, EX metadata lock,
archive, whole-directory atomic exchange. No GPU, processes or study imports.
"""
import argparse,base64,collections,copy,datetime,fcntl,gzip,hashlib,json,os,re,stat,uuid
from pathlib import Path

ALLOWED={'model_kernel_regex','semantic_family_in_primary_ROI','ROI_membership_status','ROI_membership_proof','kernel_family_alias_of','alias_of_stage_id','regex_status'}
MEMBERSHIP={'source_timed_closure_inclusion_proved','source_setup_exclusion_proved','source_untimed_audit_exclusion_proved'}
LOCK=Path('/tmp/hecbench-roofline-metadata.lock')
SOURCE=Path(__file__).read_bytes()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def encoded(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def file_bytes(p):
 p=Path(p);need(p.is_file()and not p.is_symlink(),'Missing/symlink file '+str(p));return p.read_bytes()
def json_bytes(b,p):return json.loads(gzip.decompress(b)if str(p).endswith('.gz')else b)
def binding(x):
 p=Path(x['path']);b=file_bytes(p);need(sha(b)==(x.get('sha256')or x.get('compressed_sha256')),'Binding SHA '+str(p))
 if 'bytes'in x:need(len(b)==x['bytes'],'Binding length '+str(p))
 h=x.get('uncompressed_sha256')or x.get('raw_sha256')
 if h:need(sha(gzip.decompress(b))==h,'Raw SHA '+str(p))
 return json_bytes(b,p)
def atomic_file(path,data):
 path=Path(path);temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with temp.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def tree_bytes(folder):
 need(folder.is_dir()and not folder.is_symlink(),'Canonical review tree not a regular directory');out={}
 for p in sorted(folder.iterdir()):
  need(p.is_file()and not p.is_symlink(),'Unexpected directory/symlink in review tree '+str(p));out[p.name]=file_bytes(p)
 return out

def validate(root,packets,transaction):
 study=root/'ontology/studies/v100_roofline_537';reviewdir=study/'roi_reviews';oldtree=tree_bytes(reviewdir)
 scientific_path=Path('/tmp/roofline-completion-20261021/roi_v3_root_scientific_review.json');scientific_raw=file_bytes(scientific_path);scientific=json.loads(scientific_raw)
 scientific_binding={'path':str(scientific_path),'sha256':sha(scientific_raw)}
 index=json.loads(oldtree['INDEX.json']);parents={};contexts={};ops=[];packet_receipts=[]
 for packet in packets:
  data=file_bytes(packet);doc=json.loads(data);packet_receipts.append({'path':str(packet),'sha256':sha(data)})
  need(scientific['packets'].get(str(packet))==sha(data),'Packet not covered by exact root scientific review')
  need(doc['sample_phase_still_unproved']is True and doc['elapsed_ROI_aggregate_claim']is False and doc['causal_claim']is False,'Packet promoted phase/ROI/causal facts')
  capturemap={(x['benchmark'],x['model']):x for x in doc['selected_capture_bindings']};need(len(capturemap)==len(doc['selected_capture_bindings']),'Duplicate selected binding')
  for patch in doc['patches']:
   key=(patch['record_key']['benchmark'],patch['record_key']['model']);need(key not in contexts,'Overlapping model packet '+str(key));need(key[1]in ('cuda','cuda_jl','ka','jacc'),'Unknown model')
   info=patch['parent_review'];p=Path(info['path']);need(p.parent==reviewdir and p.name.endswith('.json.gz'),'Parent escapes canonical review tree')
   expected=(info['compressed_sha256'],info.get('uncompressed_sha256')or info.get('raw_sha256'))
   if p.name in parents:need(parents[p.name]['expected']==expected,'Conflicting initial parent SHA')
   else:parents[p.name]={'expected':expected}
   contexts[key]={'patch':patch,'binding':capturemap[key],'parent':p.name}
 # Validate ALL initial parent/index references before changing even in-memory stages.
 for name,p in parents.items():
  packed=oldtree[name];raw=gzip.decompress(packed);need((sha(packed),sha(raw))==p['expected'],'Initial parent drift '+name)
  doc=json.loads(raw);p.update(original=doc,new=copy.deepcopy(doc),oldpacked=packed,oldraw=raw)
  indexrows=[x for x in index['cohorts']if root/x['path']==reviewdir/name];need(len(indexrows)==1,'INDEX parent missing/duplicate')
  need(indexrows[0]['compressed_sha256']==sha(packed)and indexrows[0]['raw_sha256']==sha(raw),'INDEX parent SHA drift')
 receipts=[];source_hashes={};source_cache={};overlay_checks=[];identities=set()
 def source(z,captured):
  p=Path(z['path']).resolve();h=z.get('source_sha256')or z.get('sha256');need(h is not None,'Missing source SHA')
  if str(p)not in source_cache:source_cache[str(p)]=file_bytes(p)
  raw=source_cache[str(p)];need(sha(raw)==h and h in captured.get(str(p),set()),'Source absent/changed in selected capture '+str(p));source_hashes[str(p)]=h
  text=raw.decode();lines=text.splitlines()
  if 'line'in z and 'text'in z:need(1<=z['line']<=len(lines)and lines[z['line']-1]==z['text'],'Exact callsite line mismatch '+str(p))
  if 'context'in z:
   start=z['context_start_line'];need(start>=1,'Context line');need('\n'.join(lines[start-1:]).startswith(z['context']),'Quoted context bytes mismatch '+str(p))
  if 'exact_lines'in z:need('\n'.join(lines[z['start_line']-1:z['end_line']])==z['exact_lines'],'Exact line span mismatch')
  for label in ('timer_lines','arithmetic_branch_atomic_lines','line_evidence'):
   for row in z.get(label,[]):
    need(isinstance(row,dict)and 1<=row['line']<=len(lines)and lines[row['line']-1]==row['text'],'Quoted source line mismatch '+str(p))
  if 'snippet'in z:
   start,end=z['start_byte'],z['end_byte'];need(isinstance(start,int)and isinstance(end,int)and 0<=start<=end<=len(raw),'Snippet byte interval')
   need(raw[start:end].decode()==z['snippet'],'Complete byte snippet mismatch '+str(p))
   if 'start_line'in z:need(raw[:start].count(b'\n')+1==z['start_line'],'Snippet start line mismatch')
   if 'end_line'in z:need(z['end_line']in {raw[:end].count(b'\n')+1,raw[:max(start,end-1)].count(b'\n')+1},'Snippet end line convention mismatch')
  return {'path':str(p),'sha256':h}
 def member(folder,manifest,name):
  entries=[e for e in manifest['files']if e['original_name']==name];need(len(entries)==1,'Manifest member missing/duplicate '+name);e=entries[0];p=root/e['path'];need(p.parent==folder,'Manifest member outside selected folder')
  packed=file_bytes(p);raw=gzip.decompress(packed);need(sha(packed)==e['compressed_sha256']and len(packed)==e['compressed_bytes']and sha(raw)==e['uncompressed_sha256']and len(raw)==e['uncompressed_bytes'],'Frozen member mismatch '+name);return json.loads(raw),raw
 for key,c in contexts.items():
  b,m=key;patch=c['patch'];doc=parents[c['parent']]['original'];rows=[x for x in doc['records']if(x['benchmark'],x['model'])==key];need(len(rows)==1,'Initial record not unique');record=rows[0]
  need(canon(record)==patch['expected_original_record_canonical_sha256'],'Initial record SHA drift '+str(key));need(doc['records'][patch['record_index']]==record,'Record index drift')
  cap=c['binding'];plan=binding(cap['plan']);derived=binding(cap.get('derivation')or cap['derived']);manifest=binding(cap['manifest']);folder=Path(cap['plan']['path']).parent
  need(folder.resolve().is_relative_to((study/'profiles').resolve()),'Selected profile outside study')
  actual_plan,planraw=member(folder,manifest,'plan.json');prov,provraw=member(folder,manifest,'profile_provenance.json');actual_derived,derivedraw=member(folder,manifest,'derived.json');need(plan==actual_plan and derived==actual_derived,'Packet/member discrepancy')
  if 'provenance'in cap:need(binding(cap['provenance'])==prov,'Provenance binding mismatch')
  need(prov['status']=='profile_process_completed'and prov.get('profile',{}).get('returncode')==0 and prov.get('profile',{}).get('stop_reason')is None,'Failed/unfinished source parent')
  need(prov['plan_sha256']==sha(planraw),'Plan/provenance join')
  ledgerpath=study/'benchmarks'/(b+'.json');ledgerraw=file_bytes(ledgerpath);ledger=json.loads(ledgerraw);state=ledger['models'][m];contract=ledger['accepted_named_contract']
  need(state['profiling_status']=='profile_process_completed'and str(folder.relative_to(root))in state['profile_ids'],'Selected capture drift '+str(key))
  need(Path(root/state['kernel_observations']['path']).parent==folder,'Selected observation folder drift')
  need(state['profile_provenance_sha256']==sha(provraw),'Selected provenance SHA drift')
  for k,ck in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]:need(plan[k]==contract[ck],'Selected contract join '+k)
  need(record['configuration_id']==plan['configuration_id']==patch['record_key']['configuration_id'],'Record CID mismatch')
  need(record['reference_result']['sha256']==plan['reference_result_sha256']and record['raw_config_sha256']==plan['raw_config_sha256'],'Candidate contract mismatch')
  for n,k in [('result.json','reference_result_sha256'),('config.json','raw_config_sha256')]:need(sha(gzip.decompress(file_bytes(study/'selected_contracts'/b/(n+'.gz'))))==plan[k],'Frozen selected contract mismatch')
  captured=collections.defaultdict(set)
  for mapping in [plan['execution_bindings'],prov.get('additional_current_profile_input_bindings',{})]:
   for p,h in mapping.items():captured[str(Path(p).resolve())].add(h)
  c.update(plan=plan,derived=derived,captured=captured)
  checked_sources=[]
  # Actual AMG executed-source text survived in the immutable selected PLAN.
  # It is never reconstructed at the absent historical scratch path.
  def overlay(items):
   for e in items:
    need(b=='amgmk','Unexpected retained-source overlay claim')
    matches=[p for p in plan.get('overlay_provenance',{}).get('profile_source_patches',[])if p['copy']==e['actual_executed_source_path']];need(len(matches)==1,'Retained AMG overlay text missing')
    p=matches[0];raw=p['text'].encode();need(e['retained_text']==p['text']and sha(raw)==p['copy_sha256']==e['actual_executed_sha256'],'Retained AMG copy byte/SHA mismatch')
    need(p['copy_sha256']in captured[str(Path(p['copy']).resolve())]and p['original_sha256']in captured[str(Path(p['original']).resolve())],'AMG original/copy absent from actual bindings')
    need(p['original']==e['original_bound_path']and p['original_sha256']==e['original_sha256'],'AMG original join')
    original=file_bytes(p['original']).decode();need(sha(original.encode())==p['original_sha256'],'AMG original source drift')
    # Reproduce only the old documented path-only transform, IN MEMORY.
    rawdir=str(Path(plan['scratch_cwd'])/'profile_outputs/raw');parent=root/'src/amgmk-julia'
    if Path(p['original']).name=='study_native.cu':
     changes=[('#include "../amgmk-cuda/main.cu"','#include '+json.dumps(str(root/'src/amgmk-cuda/main.cu'))),('#include "study_params.h"','#include '+json.dumps(str(parent/'study_params.h'))),('auto rawdir=root/".porting-state/porting_logs/amgmk-julia-shared/raw/cuda";','auto rawdir=std::filesystem::path('+json.dumps(rawdir)+');')]
    elif Path(p['original']).name=='study_common.jl':
     old='rawdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","amgmk-julia-shared","raw",model);mkpath(rawdir)';changes=[(old,'rawdir='+json.dumps(rawdir)+';mkpath(rawdir)')]
     revised=original.replace(old,changes[0][1])
     for line in revised.splitlines():
      if '@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(parent)))))
    else:
     old='include(joinpath(@__DIR__,"..","study_common.jl"))';changes=[(old,'include('+json.dumps(str(Path(p['copy']).parent/'study_common.jl'))+')')]
     for line in original.replace(old,'').splitlines():
      if '@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(Path(p['original']).parent)))))
    rebuilt=original
    for old,new in changes:need(rebuilt.count(old)==1,'AMG path-only rule ambiguous');rebuilt=rebuilt.replace(old,new,1)
    need(rebuilt==p['text'],'AMG overlay changes more than reviewed paths')
    overlay_checks.append({'benchmark':b,'model':m,'copy':p['copy'],'copy_sha256':p['copy_sha256'],'retained_plan_sha256':sha(planraw),'exact_path_only_transform_proved':True,'historical_copy_recreated':False})
  overlay(cap.get('captured_overlay_source_evidence',[]))
  newrecord=next(x for x in parents[c['parent']]['new']['records']if(x['benchmark'],x['model'])==key)
  for operation in patch['stage_patches']:
   sid=operation['stage_id'];identity=(*key,sid);need(identity not in identities,'Overlapping stage update');identities.add(identity)
   oldstage=next(x for x in record['stages']if x['stage_id']==sid);target=next(x for x in newrecord['stages']if x['stage_id']==sid);need(record['stages'][operation['stage_index']]==oldstage,'Stage index drift')
   need(canon(oldstage)==operation['expected_original_stage_canonical_sha256'],'Initial stage SHA drift '+sid)
   updates=operation['updates'];need(set(updates)<=ALLOWED,'Stage fields outside whitelist')
   alias=operation.get('alias_only',False);proof=updates.get('ROI_membership_proof')
   if alias:need(updates.get('model_kernel_regex','bad')is None and not proof and 'semantic_family_in_primary_ROI'not in updates,'Pure alias changed membership')
   else:
    need(proof and updates.get('ROI_membership_status')in MEMBERSHIP,'Unproved membership status')
    need(updates.get('semantic_family_in_primary_ROI')is(updates['ROI_membership_status']=='source_timed_closure_inclusion_proved'),'Membership Boolean/status disagree')
    for label in ('selected_result','selected_config','selected_counter_derivation','selected_profile_manifest'):
     if label in proof:binding(proof[label])
    srcs=[]
    if 'exact_stage_callsite'in proof:srcs=[proof['exact_stage_callsite']]
    srcs+=proof.get('timed_closure_boundaries',[])+proof.get('evidence',[])
    need(srcs,'No source-bound proof evidence')
    for item in srcs:checked_sources.append(source(item,captured))
    overlay(proof.get('captured_overlay_source_evidence',[]))
    need(proof.get('sample_in_primary_timed_invocation')in (None,False),'Proof promotes actual sample phase')
    need(proof.get('retained_trial')is None,'Proof invents retained trial')
    need(proof.get('whole_elapsed_ROI_claim',False)is False and proof.get('complete_ROI',False)is False,'Proof promotes whole ROI')
    need(proof.get('causal_bottleneck','unknown')in ('unknown',None),'Proof promotes causal claim')
   for item in operation.get('selector_evidence',[]):checked_sources.append(source(item,captured))
   if updates.get('model_kernel_regex')is not None:re.compile(updates['model_kernel_regex'])
   before=copy.deepcopy(target);target.update(copy.deepcopy(updates));need({k:v for k,v in before.items()if k not in updates}=={k:v for k,v in target.items()if k not in updates},'Unspecified stage fields changed')
   ops.append({'benchmark':b,'model':m,'stage_id':sid,'alias_only':alias,'inclusion':updates.get('semantic_family_in_primary_ROI'),'null_selector':updates.get('model_kernel_regex','unchanged')is None,'alias_target':updates.get('kernel_family_alias_of')or updates.get('alias_of_stage_id'),'old_stage_sha256':canon(oldstage),'new_stage_sha256':canon(target)})
  # Alias targets and all saved demangled identities remain unambiguous.
  for op in [x for x in ops if(x['benchmark'],x['model'])==key and x['alias_target']]:need(any(x['stage_id']==op['alias_target']for x in newrecord['stages']),'Missing alias target')
  matches=0
  for observation in derived['records']:
   stages=[x for x in newrecord['stages']if x.get('model_kernel_regex')and re.search(x['model_kernel_regex'],observation['identity']['Kernel Name'])];need(len(stages)<=1,'Ambiguous saved family match '+str(key));matches+=sum(x.get('semantic_family_in_primary_ROI')is True and bool(x.get('ROI_membership_proof'))for x in stages)
  need(matches>0,'No promoted stage matches saved observations')
  receipts.append({'benchmark':b,'model':m,'selected_profile':str(folder.relative_to(root)),'plan_sha256':sha(planraw),'provenance_sha256':sha(provraw),'derived_sha256':sha(derivedraw),'config_sha256':plan['raw_config_sha256'],'result_sha256':plan['reference_result_sha256'],'source_count':len({x['path']for x in checked_sources}),'saved_inclusion_matches':matches})
 # Preserve every other record, stage field, phase/causal flag and array order.
 proposed=dict(oldtree)
 for name,p in parents.items():
  for old,new in zip(p['original']['records'],p['new']['records']):need({k:v for k,v in old.items()if k!='stages'}=={k:v for k,v in new.items()if k!='stages'},'Record-wide phase/ROI/causal metadata changed')
  p['newraw']=encoded(p['new']);p['newpacked']=gzip.compress(p['newraw'],mtime=0);proposed[name]=p['newpacked']
 for row in index['cohorts']:
  name=Path(row['path']).name
  if name in parents:row.update(raw_sha256=sha(parents[name]['newraw']),compressed_sha256=sha(parents[name]['newpacked']))
 index['enrichment_receipt']='ontology/studies/v100_roofline_537/roi_reviews/enrichment_transaction.json';proposed['INDEX.json']=encoded(index)
 counts={'model_records':len(contexts),'names':len({b for b,m in contexts}),'inclusion_declarations':sum(x['inclusion']is True for x in ops),'exclusions':sum(x['inclusion']is False for x in ops),'pure_alias_nulls':sum(x['alias_only']for x in ops),'inclusion_alias_nulls':sum(x['inclusion']is True and x['null_selector']for x in ops),'non_null_inclusion_stages':sum(x['inclusion']is True and not x['null_selector']for x in ops)}
 need(counts['model_records']==56 and counts['names']==14 and counts['inclusion_declarations']==124,'Expected three-packet population not validated')
 receipt={'schema_version':1,'transaction':transaction,'status':'source_family_reviews_adopted_pending_fresh_annotation_integrity_audit','input_packets':packet_receipts,'root_scientific_review':scientific_binding,'script_sha256':sha(SOURCE),'counts':counts,'changes':ops,'selected_capture_checks':receipts,'retained_AMG_overlay_checks':overlay_checks,'sample_phase':'Unknown; source inclusion is not actual timed-invocation proof.','whole_elapsed_ROI_claim':False,'causal_claim':False,'GPU_build_tests':0}
 proposed['enrichment_transaction.json']=encoded(receipt)
 for path,h in source_hashes.items():need(sha(file_bytes(path))==h,'Source changed during validation')
 need(file_bytes(scientific_path)==scientific_raw,'Root scientific review changed during validation')
 need(tree_bytes(reviewdir)==oldtree,'Canonical review directory changed during validation')
 plan={'schema_version':1,'transaction':transaction,'source_root':str(root),'script_sha256':sha(SOURCE),'input_packets':packet_receipts,'root_scientific_review':scientific_binding,'counts':counts,'old_tree':{n:sha(b)for n,b in oldtree.items()},'new_tree':{n:sha(b)for n,b in proposed.items()},'commit_method':'Linux renameat2 RENAME_EXCHANGE whole roi_reviews directory; no sequential fallback','source_bindings':source_hashes}
 return plan,receipt,oldtree,proposed

def exchange(a,b):
 # No execution occurs in dry-run. Root apply only; no privilege or processes.
 import ctypes
 libc=ctypes.CDLL(None,use_errno=True);fn=getattr(libc,'renameat2',None);need(fn is not None,'renameat2 unavailable; no sequential replacement fallback')
 fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 if fn(-100,os.fsencode(a),-100,os.fsencode(b),2)!=0:raise OSError(ctypes.get_errno(),'Atomic directory exchange failed')

def sync_directory(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--apply',action='store_true');ap.add_argument('--approval-plan',type=Path);ap.add_argument('--approval-plan-sha256');ap.add_argument('packets',nargs='+',type=Path);a=ap.parse_args()
 root=a.source_root.resolve();out=a.output_root.resolve();need(out.is_relative_to('/tmp')and out!=Path('/tmp')and not out.exists(),'Fresh private /tmp report directory required');out.mkdir(parents=True,mode=0o700)
 approval=None
 if a.apply:
  need(a.approval_plan is not None and a.approval_plan_sha256,'Root apply requires exact reviewed dry-run plan')
  raw=file_bytes(a.approval_plan);need(sha(raw)==a.approval_plan_sha256,'Approval-plan SHA');approval=json.loads(raw)
 token=approval['transaction']if approval else'roi-v3-'+uuid.uuid4().hex[:16];need(re.fullmatch(r'roi-v3-[0-9a-f]{16}',token)is not None,'Transaction token')
 result={'started_at':utc(),'apply_requested':a.apply,'shared_mutations':False,'transaction':token,'script_sha256':sha(SOURCE)}
 try:
  with LOCK.open('r')as lock:
   fcntl.flock(lock,fcntl.LOCK_EX if a.apply else fcntl.LOCK_SH)
   plan,receipt,old,new=validate(root,[x.resolve()for x in a.packets],token)
   atomic_file(out/'merge_plan.json',encoded(plan));atomic_file(out/'validated_receipt.json',encoded(receipt))
   if not a.apply:
    result.update(status='dry_run_validated_no_shared_writes',counts=plan['counts'],merge_plan_sha256=sha(encoded(plan)));atomic_file(out/'validation_report.json',encoded(result));print(json.dumps(result));return 0
   need(plan==approval,'Reviewed plan no longer matches complete fresh validation')
   study=root/'ontology/studies/v100_roofline_537';target=study/'roi_reviews';stage=study/('.'+token+'-staged');history=study/'roi_review_history';need(not stage.exists(),'Refuse existing transaction staging path')
   # Stage whole regular-file tree. Existing symlinks/subdirectories were rejected.
   stage.mkdir(mode=stat.S_IMODE(target.stat().st_mode));history.mkdir(exist_ok=True)
   for name,data in new.items():atomic_file(stage/name,data)
   need(tree_bytes(stage)==new,'Staged tree mismatch');need(tree_bytes(target)==old,'Initial tree drift before commit')
   for name,data in old.items():
    if name not in new or new[name]!=data:
     suffix='.json.gz'if name.endswith('.json.gz')else'.'+name
     dest=history/(sha(data)+suffix)
     if dest.exists():need(file_bytes(dest)==data,'History archive collision')
     else:atomic_file(dest,data)
   journal={'transaction':token,'state':'prepared_before_atomic_exchange','target':str(target),'staged':str(stage),'old_tree':plan['old_tree'],'new_tree':plan['new_tree']};atomic_file(out/'transaction_journal.json',encoded(journal))
   committed=False
   try:
    exchange(target,stage);committed=True
    sync_directory(target.parent)
    need(tree_bytes(target)==new and tree_bytes(stage)==old,'Post-exchange tree mismatch')
    journal['state']='committed_atomic_directory_exchange';atomic_file(out/'transaction_journal.json',encoded(journal))
    result.update(status='applied_atomic_directory_exchange_pending_annotation_audit',shared_mutations=True,counts=plan['counts'],rollback_tree=str(stage),merge_plan_sha256=sha(encoded(plan)))
    atomic_file(out/'validation_report.json',encoded(result))
   except BaseException:
    if committed:
     exchange(target,stage);sync_directory(target.parent);need(tree_bytes(target)==old,'Rollback verification failed');journal['state']='rolled_back_complete_original_tree';atomic_file(out/'transaction_journal.json',encoded(journal))
    raise
   fcntl.flock(lock,fcntl.LOCK_UN)
  print(json.dumps(result));return 0
 except BaseException as e:
  result.update(status='failed_no_admission',error=repr(e));atomic_file(out/'validation_report.json',encoded(result));print(json.dumps(result));return 1
if __name__=='__main__':raise SystemExit(main())
