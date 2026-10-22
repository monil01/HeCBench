import json,gzip,hashlib,fcntl,datetime,re
from pathlib import Path
O=Path(__file__).parent;W=Path('/home/imo/HeCBench_julia');p=O/'proposed_stage_patches.json';d=json.loads(p.read_bytes());H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();C=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest();checks=[]
def ck(label,yes):
 checks.append({'check':label,'passed':bool(yes)})
 assert yes,label
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 index=d['parent_INDEX'];ck('INDEX exact bytes',H(index['path'])==index['sha256']);idx=json.loads(Path(index['path']).read_bytes());ir={str((W/r['path']).resolve()):r for r in idx['cohorts']};parents={}
 for path,e in d['parent_reviews'].items():
  pb=Path(path).read_bytes();raw=gzip.decompress(pb);ck('parent compressed '+path,hashlib.sha256(pb).hexdigest()==e['compressed_sha256']);ck('parent raw '+path,hashlib.sha256(raw).hexdigest()==e['uncompressed_sha256']);ck('INDEX parent join '+path,ir[path]['compressed_sha256']==e['compressed_sha256']and ir[path]['raw_sha256']==e['uncompressed_sha256']);parents[path]=json.loads(raw)
 for e in d['sources']:ck('source '+e['path'],H(e['path'])==e['sha256'])
 caps={}
 for r in d['selected_capture_bindings']:
  for k in ['ledger','manifest','plan','derivation']:ck(k+' '+r['benchmark']+'/'+r['model'],H(r[k]['path'])==r[k]['sha256'])
  plan=json.loads(gzip.decompress(Path(r['plan']['path']).read_bytes()));ledger=json.loads(Path(r['ledger']['path']).read_bytes());a=ledger['accepted_named_contract'];ck('canonical accepted CID/config/result '+r['benchmark']+'/'+r['model'],a['configuration_id']==plan['configuration_id']and a['raw_config_sha256']==plan['raw_config_sha256']and a['result_sha256']==plan['reference_result_sha256']);caps[(r['benchmark'],r['model'])]=r
 for r in d['patches']:
  k=r['record_key'];old=parents[r['parent_review']['path']]['records'][r['record_index']];ck('record hash '+str(k),C(old)==r['expected_original_record_canonical_sha256']);ck('record key '+str(k),all(old[x]==v for x,v in k.items()));new=[dict(x)for x in old['stages']]
  for sp in r['stage_patches']:
   oldstage=old['stages'][sp['stage_index']];ck('stage hash '+sp['stage_id'],oldstage['stage_id']==sp['stage_id']and C(oldstage)==sp['expected_original_stage_canonical_sha256']);new[sp['stage_index']].update(sp['updates'])
  for key,value in r['unchanged_record_qualifications'].items():ck('unchanged '+key+' '+str(k),old[key]==value)
  ob=json.loads(gzip.decompress(Path(caps[(k['benchmark'],k['model'])]['derivation']['path']).read_bytes()))['records'];matches=[]
  for o in ob:
   hits=[s['stage_id']for s in new if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],o['identity']['Kernel Name'])];ck('unique savedfamily '+str(k),len(hits)<=1);ck('matched family included '+str(k),all(s['semantic_family_in_primary_ROI']is True for s in new if s['stage_id']in hits));matches.append({'identity':o['identity'],'matched_stage_ids':hits})
  ck('positive savedfamily '+str(k),any(x['matched_stage_ids']for x in matches));ck('savedmatch equality '+str(k),matches==[{x:v[x]for x in ['identity','matched_stage_ids']}for v in r['selected_observation_matching']])
 le=d['legacy_authority_exclusion_check'];ck('legacy compressed',H(le['path'])==le['compressed_sha256']);ck('legacy receipt',H(le['receipt']['path'])==le['receipt']['sha256']);ck('no legacy names',not any(r['benchmark']in {x['record_key']['benchmark']for x in d['patches']}for r in json.loads(gzip.decompress(Path(le['path']).read_bytes()))['records']))
 fcntl.flock(lock,fcntl.LOCK_UN)
(O/'final_locked_revalidation.json').write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_bindings_stable':True,'checks':checks},indent=2)+'\n')
st=[s for r in d['patches']for s in r['stage_patches']];inc=sum(s['updates']['semantic_family_in_primary_ROI']for s in st);obs=[x for r in d['patches']for x in r['selected_observation_matching']];hit=sum(bool(x['matched_stage_ids'])for x in obs)
summary={'models_proposed':len(d['patches']),'names':len({r['record_key']['benchmark']for r in d['patches']}),'stages':len(st),'included':inc,'excluded':len(st)-inc,'saved_observations':len(obs),'matched':hit,'unassigned':len(obs)-hit,'withheld':[{'benchmark':r['benchmark'],'model':r['model'],'reason':r['reason']}for r in d['withheld_models']],'checks':len(checks),'proposal_sha256':H(p),'callable_value_approvals':[{'record_key':r['record_key'],'stage_id':s['stage_id'],'resolved_callable_candidates':s['updates']['resolved_callable_candidates']}for r in d['patches']for s in r['stage_patches']if'resolved_callable_candidates'in s['updates']]};(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='callable_value_approvals'}))
