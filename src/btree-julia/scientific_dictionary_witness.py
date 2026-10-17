"""Independent exact integer dictionary oracle for saved tree artifacts; no GPU run."""
import array,json,sys
from pathlib import Path
resultpath=Path(sys.argv[1]).resolve();result=json.loads(resultpath.read_text());prefix=[];expected=[];cases=[]
for n in (256,4096):
 keys=[(i*197+53)%n for i in range(n)];values=[(k*104729+97)%10000019+11 for k in keys];queries=[(i*313+19)%(2*n) for i in range(2*n)];mapping=dict(zip(keys,values))
 assert len(mapping)==n and len(set(queries))==2*n
 prefix.extend(keys+values+queries+[n,2*n,1<<18,256]);wantq=[mapping.get(q,0)for q in queries];wantpairs=[v for pair in sorted(mapping.items())for v in pair];expected.extend(wantq+wantpairs);cases.append({'n':n,'present_queries':sum(q in mapping for q in queries),'absent_queries':sum(q not in mapping for q in queries),'complete_dictionary_pairs':n,'zero_value_not_used':all(v>0 for v in values)})
report={'oracle':'Independent Python exact integer dictionary, exhaustive present/absent queries and sorted complete inserted key/value pairs. Does not traverse or imitate GPU tree splits.','input_prefix_values':len(prefix),'output_values':len(expected),'cases':cases,'trials':[],'scope':'Semantic dictionary and query correctness, not identical physical tree shape, allocator IDs, root depth or a formal concurrency proof.'}
for model,spec in result['models'].items():
 for trial in spec.get('trials',[]):
  path=Path(trial['output_artifact'].get('saved_path',trial['output_artifact']['path']));path=path if path.is_absolute()else resultpath.parent/path
  data=array.array('I');data.frombytes(path.read_bytes())
  if sys.byteorder!='little':data.byteswap()
  assert list(data[:len(prefix)])==prefix,(model,trial['trial'],'inputs')
  assert list(data[len(prefix):])==expected,(model,trial['trial'],'dictionary')
  report['trials'].append({'model':model,'trial':trial['trial'],'exact_prefix':True,'all_queries_and_complete_dictionary_exact':True})
out=resultpath.parent/'scientific_dictionary_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
