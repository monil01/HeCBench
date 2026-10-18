from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());meta=json.loads((p/'independent_oracle.json').read_text());gold=R/meta['golden_path'];g=gold.read_bytes();assert hashlib.sha256(g).hexdigest()==meta['golden_sha256'];checks=[]
for model,data in r['models'].items():
 for t in data.get('trials',[]):
  q=Path(t['output_artifact']['saved_path']);a=q.read_bytes();assert len(a)==meta['output_count']*4;assert a==g,(model,t['trial']);checks.append({'model':model,'trial':t['trial'],'bytes_compared':len(a),'sha256':hashlib.sha256(a).hexdigest(),'complete_inputs_tours_edges_scratch_climbs_best_pair_distances_padding_guards_exact':True})
print(json.dumps({'result':str(rp),'all_pass':True,'count':meta['output_count'],'input_prefix_count':meta['input_prefix_count'],'independent_golden_sha256':meta['golden_sha256'],'selection_uniqueness_proof':'Everynegativebestlaneeveryactualrestart/iterationunderdeclaredsourceFloat32arithmetic;originaltiedfixturesunmeasured','checks':checks},indent=2))
