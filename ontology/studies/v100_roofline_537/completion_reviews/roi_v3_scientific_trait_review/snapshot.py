from pathlib import Path
import json,hashlib,fcntl
R=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
files={};data={};raws={}
def take(p):
 p=Path(p);b=p.read_bytes();key=str(p.relative_to(R))if p.is_relative_to(R)else str(p);files[key]={'sha256':sha(b),'bytes':len(b)};raws[key]=b;return json.loads(b)if p.suffix=='.json'else b.decode()
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 a=take(R/'ontology/studies/v100_roofline_537/completion_reviews/roi_membership_root_merge_v3/root_adoption.json');names=sorted({c['benchmark']for c in a['changes']})
 for n in names:
  d=take(R/f'ontology/studies/julia_abstractions_537/domains/{n}.json');impl={m:take(R/f'ontology/studies/julia_abstractions_537/implementations/{n}/{m}.json')for m in ['cuda_jl','ka','jacc']};l=take(R/f'ontology/studies/v100_roofline_537/benchmarks/{n}.json');base=take(R/f'ontology/benchmarks/{n}.yaml');data[n]={'scientific_intent':d.get('scientific_intent'),'implementations':{m:{'normalized':j.get('implementation_ontology_normalized'),'shared_traits':j.get('shared_benchmark_traits'),'configurations':j.get('configurations')}for m,j in impl.items()},'selected_contract':l.get('accepted_named_contract'),'base_yaml':base}
 for packet in a['input_packets']:take(packet['path'])
 for f in ['ontology/tools/generate_ontology_report.py','ontology/tools/generate_relationship_graph.py','ontology/tools/julia_graph/graph.js']:take(R/f)
 fcntl.flock(lock,fcntl.LOCK_UN)
for key,b in raws.items():
 p=O/'inputs'/key.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
(O/'snapshot_bindings.json').write_text(json.dumps(files,indent=2)+'\n');(O/'extracted.json').write_text(json.dumps(data,indent=2)+'\n')
for n,d in data.items():
 print('\n###',n)
 print('BASE',d['base_yaml'][:1550])
 print('TRAITS',json.dumps(d['implementations']['cuda_jl']['normalized'],ensure_ascii=False)[:2800])
 print('SCOPE',json.dumps(d['implementations']['cuda_jl']['shared_traits'],ensure_ascii=False)[:1000])
