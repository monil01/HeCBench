import json,hashlib,pathlib,datetime
ROOT=pathlib.Path('/home/imo/HeCBench_julia'); OUT=pathlib.Path(__file__).parent
names=['blas-fp4gemm','blas-fp8gemm','blas-mxfp6gemm','blas-mxfp8gemm','twell']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def resolve(s):
 p=pathlib.Path(s);return p if p.is_absolute() else ROOT/p
checks=[];seen=set();docs=[]
def visit_doc(p):
 p=p.resolve()
 if p in seen:return
 seen.add(p)
 try:d=json.loads(p.read_text())
 except Exception:return
 docs.append({'path':str(p),'sha256':digest(p)})
 def walk(x,loc=''):
  if isinstance(x,dict):
   path=x.get('path') or x.get('resolved_path'); sha=x.get('sha256')
   if isinstance(path,str) and isinstance(sha,str) and len(sha)==64:
    q=resolve(path); rec={'declared_in':str(p),'json_location':loc,'path':str(q),'expected_sha256':sha,'exists':q.is_file()}
    if q.is_file():rec.update(actual_sha256=digest(q));rec['match']=rec['actual_sha256']==sha
    else:rec['match']=False
    checks.append(rec)
    if q.suffix=='.json' and q.is_file():visit_doc(q)
   for k,v in x.items():walk(v,loc+'/'+k)
  elif isinstance(x,list):
   for i,v in enumerate(x):walk(v,loc+'/'+str(i))
 walk(d)
rows=[]
for name in names:
 parent=ROOT/'src'/f'{name}-julia'; review=parent/'limitation_review.json';visit_doc(review)
 for p in parent.glob('*review*.json'):visit_doc(p)
 statuses=[]
 for model in ['cuda_jl','ka','jacc']:
  p=ROOT/'.porting-state/portable_julia/status_300'/name/f'{model}.json';visit_doc(p);d=json.loads(p.read_text())
  statuses.append({'model':model,'path':str(p),'sha256':digest(p),'status':d.get('status'),'terminal_adjudication':d.get('terminal_adjudication'),'review_completed':d.get('repair_or_limitation_review_completed'),'framework_unsupported':d.get('framework_unsupported'),'new_translation_attempts':d.get('new_translation_attempts',d.get('target_translation_attempted'))})
 rows.append({'benchmark':name,'statuses':statuses})
for path in ['src/twell-julia/domain_snippet.json','src/twell-julia/implementation_traits.json','.porting-state/portable_julia/traits/twell.json','.porting-state/portable_julia/scientific_intent_300/twell.json','ontology/studies/julia_abstractions_537/domains/twell.json']:
 p=ROOT/path;visit_doc(p)
 d=json.loads(p.read_text());i=d.get('scientific_intent',d)
 rows[-1].setdefault('domain_reviews',[]).append({'path':path,'sha256':digest(p),'review_status':i.get('review_status'),'primary_family':i.get('primary_family'),'mathematical_operation':i.get('mathematical_operation')})
summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only saved evidence; no GPU, no source/state/package/Git changes. Hash checks cover explicit path/sha256 and resolved_path/sha256 pairs reachable from listed review/status/domain JSON, not arbitrary textual references.','cases':rows,'documents':docs,'evidence_checks':checks,'counts':{'benchmarks':len(rows),'model_statuses':sum(len(r['statuses']) for r in rows),'explicit_hash_bindings':len(checks),'matching':sum(c['match'] for c in checks),'missing':sum(not c['exists'] for c in checks),'mismatched':sum(c['exists'] and not c['match'] for c in checks)}}
(OUT/'review.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['counts']))
for c in checks:
 if not c['match']: print(json.dumps(c))
print('TWELL',json.dumps(rows[-1]['domain_reviews']))
