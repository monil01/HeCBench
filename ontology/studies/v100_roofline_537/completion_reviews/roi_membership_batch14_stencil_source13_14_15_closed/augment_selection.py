from pathlib import Path
import json,gzip,hashlib,fcntl,shutil
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).parent;H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();selection=O/'selection.json';rr=O/'selected_records.json.gz';shutil.copy2(selection,O/'selection_001_initial.json');shutil.copy2(rr,O/'selected_records_001_initial.json.gz');d=json.loads(selection.read_bytes());rows=json.loads(gzip.decompress(rr.read_bytes()))
with open('/tmp/hecbench-roofline-metadata.lock','r')as l:
 fcntl.flock(l,fcntl.LOCK_SH);assert H(S/'roi_reviews/INDEX.json')==d['INDEX']['sha256'];idx=json.loads((S/'roi_reviews/INDEX.json').read_bytes());extra=[]
 for p in idx['cohorts']:
  path=W/p['path'];doc=json.loads(gzip.decompress(path.read_bytes()))
  for i,r in enumerate(doc['records']):
   if r['benchmark']=='reverse2D':
    assert not any(s.get('ROI_membership_status','').startswith('source_')for s in r['stages']);m=r['model'];ledger=json.loads((S/'benchmarks/reverse2D.json').read_bytes());state=ledger['models'][m];assert state['profiling_status']=='profile_process_completed';plan=json.loads(gzip.decompress(((W/state['profile_evidence_manifest']).parent/'plan.json.gz').read_bytes()));ct=ledger['accepted_named_contract'];assert plan['configuration_id']==r['configuration_id']==ct['configuration_id']and plan['reference_result_sha256']==r['reference_result']['sha256']==ct['result_sha256']and plan['raw_config_sha256']==r['raw_config_sha256']==ct['raw_config_sha256'];extra.append({'parent':str(path),'index':i,'record':r})
 assert len(extra)==4;rows+=extra;d['selected_names'].append('reverse2D');d['eligible_requested_names']=[b for b in d['selected_names']if b!='reverse'];d['preliminary_contract_holds']=[{'benchmark':'reverse','model':m,'reason':'Canonicalconfiguration_idnull vsnamedparent/selectedprofile; result/configSHAagree, no fallback or CIDrepair.'}for m in ['cuda','cuda_jl','ka','jacc']];rr.write_bytes(gzip.compress((json.dumps(rows,indent=2)+'\n').encode(),mtime=0));d['selected_records_gzip_sha256']=H(rr);selection.write_text(json.dumps(d,indent=2)+'\n')
