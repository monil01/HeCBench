from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());meta=json.loads((p/'independent_oracle.json').read_text());gold=(R/meta['golden_path']).read_bytes();assert hashlib.sha256(gold).hexdigest()==meta['golden_sha256'];checker_sha=hashlib.sha256((p/'study_capture.py').read_bytes()).hexdigest();checks=[]
for model,d in r['models'].items():
 for t in d.get('trials',[]):
  a=Path(t['output_artifact']['saved_path']).read_bytes();assert a==gold,(model,t['trial']);q=Path(t['raw_state_index']);review=json.loads(q.read_text());assert review['all_checks_pass']and review['complete_physical_state_retained']and review['raw_output_identity_claim']is False;assert review['checker_sha256']==checker_sha;raw=Path(review['raw_path']);assert raw.stat().st_size==review['raw_bytes'];h=hashlib.sha256()
  with raw.open('rb')as f:
   for part in iter(lambda:f.read(1048576),b''):h.update(part)
  assert h.hexdigest()==review['raw_sha256'];assert len(review['cases'])==4 and sum(c['complete_voxels']for c in review['cases'])==1280
  checks.append({'model':model,'trial':t['trial'],'canonical_count':len(a)//4,'canonical_sha256':hashlib.sha256(a).hexdigest(),'all_keyed_numeric_values_exact':True,'raw_state_index':str(q),'raw_state_index_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'raw_sha256':h.hexdigest(),'raw_bytes':raw.stat().st_size,'raw_mapping_source_checks_pass':True})
print(json.dumps({'result':str(rp),'all_pass':True,'input_prefix_count':meta['input_prefix_count'],'count':meta['output_count'],'numerical_contract':'Completekeyednumericsemanticagreementnotphysicalorder','raw_output_identity_claim':False,'distinct_physical_state_hashes':len({c['raw_sha256']for c in checks}),'checks':checks},indent=2))
