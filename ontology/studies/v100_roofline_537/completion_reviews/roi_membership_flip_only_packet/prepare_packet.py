from pathlib import Path
import json,gzip,hashlib,copy,fcntl,datetime
D=Path(__file__).resolve().parent;O=Path('/tmp/roofline-completion-20261021/roi_membership_source_batch5_f');R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537'
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())
def need(ok,msg):
 if not ok:raise ValueError(msg)
def jread(p):b=Path(p).read_bytes();return json.loads(gzip.decompress(b)if str(p).endswith('.gz')else b),b
orig,origb=jread(O/'proposed_stage_patches.json');frozen,frozenb=jread(O/'selected_review_records.json.gz');patches=copy.deepcopy([x for x in orig['patches']if x['record_key']['benchmark']=='flip']);caps=copy.deepcopy([x for x in orig['selected_capture_bindings']if x['benchmark']=='flip']);need(len(patches)==len(caps)==4,'Exactlyflip4')
receipts=[];parents={};sourcepaths=set();bindings={};newrecords=[];counts={'model_records':4,'names':1,'inclusion_declarations':0,'exclusions':0,'pure_alias_nulls':0,'inclusion_alias_nulls':0,'non_null_inclusion_stages':0}
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for p in patches:
  key=p['record_key'];m=key['model'];old=next(x for x in frozen if x['record']['benchmark']=='flip'and x['record']['model']==m);path=Path(p['parent_review']['path']);doc,pb=jread(path);matches=[(i,x)for i,x in enumerate(doc['records'])if x['benchmark']=='flip'and x['model']==m];need(len(matches)==1,'Currentrecordunique');idx,record=matches[0]
  need(record==old['record']and canon(record)==p['expected_original_record_canonical_sha256'],'Target record changed; do not rebase')
  current={'path':str(path),'compressed_sha256':sha(pb),'uncompressed_sha256':sha(gzip.decompress(pb))};parents[str(path)]={k:v for k,v in current.items()if k!='path'};bindings[str(path)]=sha(pb);oldparent=copy.deepcopy(p['parent_review']);oldindex=p['record_index'];p['parent_review']=current;p['record_index']=idx
  originalpatch=next(x for x in orig['patches']if x['record_key']==key);need(p['stage_patches']==originalpatch['stage_patches'],'Stage proof/update content changed')
  cap=next(x for x in caps if x['model']==m)
  for role in ['ledger','manifest','plan','derivation','provenance']:
   x=cap[role];b=Path(x['path']).read_bytes();need(sha(b)==x['sha256'],'Originalselectedbindingchanged '+role+' '+m);bindings[x['path']]=sha(b)
  plan,_=jread(cap['plan']['path']);prov,_=jread(cap['provenance']['path']);ledger,_=jread(cap['ledger']['path']);need(ledger['models'][m]['profile_provenance_sha256']==sha(gzip.decompress(Path(cap['provenance']['path']).read_bytes())),'Selectedprofile changed');need(Path(ledger['models'][m]['kernel_observations']['path']).parent==Path(cap['plan']['path']).parent.relative_to(R),'Selectedfolder changed');need(plan['reference_result_sha256']=='803e84790e80a9bc706967cb8633c542a83eeba0f00d29eff9ddf2d9a3477e7d'and plan['raw_config_sha256']=='371c61a3cac140641d17e038c9118551a86626adc61201caf9d00b2c7f283a9c','Frozenidentity changed');need(prov['status']=='profile_process_completed'and prov['profile']['returncode']==0 and prov['profile']['stop_reason']is None,'Profileunfinished')
  for op in p['stage_patches']:
   stage=record['stages'][op['stage_index']];need(stage['stage_id']==op['stage_id']and canon(stage)==op['expected_original_stage_canonical_sha256'],'Originalstagechanged');u=op['updates'];inc=u.get('semantic_family_in_primary_ROI');null=u.get('model_kernel_regex','unchanged')is None;counts['inclusion_declarations']+=inc is True;counts['exclusions']+=inc is False;counts['pure_alias_nulls']+=op.get('alias_only',False);counts['inclusion_alias_nulls']+=inc is True and null;counts['non_null_inclusion_stages']+=inc is True and not null
   proof=u.get('ROI_membership_proof',{});items=proof.get('evidence',[])+proof.get('timed_closure_boundaries',[])+([proof['exact_stage_callsite']]if' exact_stage_callsite'in proof else[])
   if 'exact_stage_callsite'in proof:items.append(proof['exact_stage_callsite'])
   for z in items:sourcepaths.add(z['path'])
  receipts.append({'model':m,'old_parent':oldparent,'current_parent':current,'old_index':oldindex,'current_index':idx,'target_record_canonical_sha256':canon(record),'target_record_exact_object_equal':True,'entire_stage_patches_unchanged':True,'original_selected_capture_bindings_unchanged':True});newrecords.append({'parent':str(path),'index':idx,'record':record})
 sources=[copy.deepcopy(x)for x in orig['sources']if x['path']in sourcepaths]
 need({x['path']for x in sources}==sourcepaths,'Missingsourcebinding')
 for x in sources:
  b=Path(x['path']).read_bytes();need(sha(b)==x['sha256'],'Sourcechanged');bindings[x['path']]=sha(b)
 for path,h in bindings.items():need(sha(Path(path).read_bytes())==h,'Snapshotdrift')
 fcntl.flock(lock,fcntl.LOCK_UN)
packet=copy.deepcopy(orig);packet.update(scope='Flip-only4 unchanged-stage proof extraction with current-parent refresh; requires separately reviewed exact historic CID exception',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_reviews=parents,selected_capture_bindings=caps,sources=sources,patches=patches,withheld_stages=[],withheld_models=[],before_after_bindings_stable=True)
(D/'proposed_stage_patches.json').write_text(json.dumps(packet,indent=2)+'\n');(D/'selected_original_records.json.gz').write_bytes(gzip.compress((json.dumps(newrecords,indent=2)+'\n').encode(),mtime=0))
report={'status':'PRIVATE_PACKET_ROOT_REVIEW_REQUIRED_NO_ADMISSION','original_packet':{'path':str(O/'proposed_stage_patches.json'),'sha256':sha(origb)},'original_frozen_records':{'path':str(O/'selected_review_records.json.gz'),'sha256':sha(frozenb)},'expected_counts':counts,'parent_refresh_proofs':receipts,'before_after_bindings':bindings,'selected_bindings_refreshed':False,'stage_proofs_or_updates_changed':False,'canonical_contract_CID_stays_null':True,'historical_result_unchanged':True,'snapshot':'SharedmetadataLOCK_SH construction and final fullinputhash stability; rootmerge mustrecheck afterbatch6 selectedstate','scientific_scope':'FP32 named3D16cubefixture, exact reversal; native generic index/strides vs Julia specialized cubic; originalextraFP64scope unmeasured. Warmup excluded and timedrepeat sourcefamily included, actualsamplephase unknown, no wholeROI/causalclaim.','root_prerequisites':['Exact flip-only rootnote approval hashes required by private merge_flip_exact.py','Root-current scientific note includes exact packetSHA and seven counts','Current parents/source/capture joins revalidated at dryrun/apply','No concurrent sourceauthority application or script substitution'],'GPU_build_tests':0,'shared_mutations':0}
(D/'refresh_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'counts':counts,'records':receipts,'sourcefiles':len(sources)}))
