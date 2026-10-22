import json,gzip,hashlib,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N='qem qkv qtclustering quantVLLM quicksort radixsort radixsort2 rainflow'.split()
def sha(b):return hashlib.sha256(b).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 ip=S/'roi_reviews/INDEX.json';ib=ip.read_bytes();idx=json.loads(ib);lp=S/'ROI_REVIEW.json.gz';legacy=json.loads(gzip.decompress(lp.read_bytes()));excluded={r['benchmark']for r in legacy['records']};rows=[];allrows=[]
 for c in idx['cohorts']:
  p=W/c['path'];b=p.read_bytes();raw=gzip.decompress(b);assert sha(b)==c['compressed_sha256']and sha(raw)==c['raw_sha256']
  for i,r in enumerate(json.loads(raw)['records']):
   if any(s.get('ROI_membership_status','').startswith('source_')or s.get('ROI_membership_proof')for s in r['stages']):excluded.add(r['benchmark'])
   allrows.append({'parent':str(p),'index':i,'record':r})
 for dirname in ['roi_membership_source_batch11_np','roi_membership_batch11_stencil','roi_membership_source_batch12_np','roi_membership_batch12_stencil']:
  x=json.loads((O.parent/dirname/'proposed_stage_patches.json').read_bytes())
  excluded.update(p['record_key']['benchmark']for p in x['patches']);excluded.update(p['benchmark']for p in x.get('withheld_models',[]))
 assert not set(N)&excluded
 joins=[]
 for n in N:
  ledger=json.loads((S/'benchmarks'/f'{n}.json').read_bytes());ct=ledger['accepted_named_contract']
  for m in ['cuda','cuda_jl','ka','jacc']:
   st=ledger['models'][m];assert st['profiling_status']=='profile_process_completed';p=W/st['profile_evidence_manifest'];plan=json.loads(gzip.decompress((p.parent/'plan.json.gz').read_bytes()));r=[x for x in allrows if x['record']['benchmark']==n and x['record']['model']==m];assert len(r)==1;rr=r[0]['record'];assert plan['configuration_id']==ct.get('configuration_id')==rr['configuration_id'];assert plan['reference_result_sha256']==ct['result_sha256']==rr['reference_result']['sha256'];assert plan['raw_config_sha256']==ct['raw_config_sha256']==rr['raw_config_sha256'];rows+=r;joins.append({'benchmark':n,'model':m,'exact_CID_result_config_join':True})
 assert ip.read_bytes()==ib
 (O/'selection_parent_INDEX.json').write_bytes(ib);(O/'selected_review_records.json.gz').write_bytes(gzip.compress((json.dumps(rows,indent=2)+'\n').encode(),mtime=0));(O/'selection_authority_review.json').write_text(json.dumps({'names':N,'legacy':file(lp),'INDEX':file(ip),'existing_source_and_legacy_names_excluded':sorted(excluded),'all11and12_proposal_names_and_holds_excluded':True,'joins':joins},indent=2)+'\n')
 print('Selected',N,'exact joins',len(joins))
