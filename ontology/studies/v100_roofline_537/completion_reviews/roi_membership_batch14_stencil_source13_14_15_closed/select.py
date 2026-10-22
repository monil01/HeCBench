from pathlib import Path
import json,gzip,hashlib,fcntl
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).parent;H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();reserved={'qem','qkv','qtclustering','quantVLLM','quicksort','radixsort','radixsort2','rainflow'};annot=set();held=set();records={};parents={}
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);index=S/'roi_reviews/INDEX.json';ii=json.loads(index.read_bytes())
 for row in ii['cohorts']:
  p=W/row['path'];raw=gzip.decompress(p.read_bytes());assert H(p)==row['compressed_sha256']and hashlib.sha256(raw).hexdigest()==row['raw_sha256'];doc=json.loads(raw);parents[str(p)]={'compressed_sha256':H(p),'uncompressed_sha256':row['raw_sha256']}
  for i,r in enumerate(doc['records']):
   if any(st.get('ROI_membership_status','').startswith('source_')for st in r['stages']):annot.add(r['benchmark'])
   records[r['benchmark'],r['model']]={'parent':str(p),'index':i,'record':r}
 legacy=S/'ROI_REVIEW.json.gz';ld=json.loads(gzip.decompress(legacy.read_bytes()))
 for r in ld['records']:
  if any(st.get('ROI_membership_status','').startswith('source_')for st in r['stages']):annot.add(r['benchmark'])
 for root in Path('/tmp/roofline-completion-20261021').glob('*batch1*'):
  p=root/'proposed_stage_patches.json'
  if p.is_file():
   d=json.loads(p.read_bytes());held.update(x['benchmark']for x in d.get('withheld_models',[]));annot.update(x['record_key']['benchmark']for x in d.get('patches',[])if 'batch13'in str(root))
 eligible=[];excluded=[]
 for p in sorted((S/'benchmarks').glob('*.json')):
  b=p.stem
  if not b.startswith(('r','s'))or b in reserved or b in annot or b in held:continue
  d=json.loads(p.read_bytes())
  if all(d['models'][m]['profiling_status']=='profile_process_completed'and d['models'][m].get('sampled_kernel_count',0)>0 and(b,m)in records for m in ['cuda','cuda_jl','ka','jacc']):eligible.append(b)
 names=eligible[:8];rows=[records[b,m]for b in names for m in ['cuda','cuda_jl','ka','jacc']];(O/'selected_records.json.gz').write_bytes(gzip.compress((json.dumps(rows,indent=2)+'\n').encode(),mtime=0));(O/'selection.json').write_text(json.dumps({'selected_names':names,'remaining_eligible':eligible[8:],'excluded_source_annotated_names':sorted(annot),'excluded_knownheld_names':sorted(held),'cold13_reserved':sorted(reserved),'INDEX':{'path':str(index),'sha256':H(index)},'legacy':{'path':str(legacy),'sha256':H(legacy)},'selected_records_gzip_sha256':H(O/'selected_records.json.gz')},indent=2)+'\n');print(names)
 fcntl.flock(lock,fcntl.LOCK_UN)
