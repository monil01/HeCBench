from pathlib import Path
import json,hashlib,datetime,subprocess
R=Path('/home/imo/HeCBench_julia');P=R/'src/_portable_study/completion537_20261020';P.mkdir(parents=True,exist_ok=True);receipts={}
for filename in ['commit537_outcomes.json','combined537_outcomes.json']:
 path=Path('/tmp')/filename
 if not path.exists():continue
 data=json.loads(path.read_text());(P/filename).write_bytes(path.read_bytes())
 for row in data:
  for item in row['phases']:
   if item['returncode']==0:receipts.setdefault(row['benchmark'],{})[item['phase']]=item['commit']
for phase in ['shared','ka','jacc']:
 f=Path('/tmp')/f'minibude_commit_{phase}.log'
 if f.exists():
  for line in f.read_text().splitlines():
   try:x=json.loads(line)
   except ValueError:continue
   if x.get('benchmark')=='minibude' and 'phase'in x:receipts.setdefault('minibude',{})[x['phase']]=x['commit']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();joined=[]
for bench,phases in receipts.items():
 for m,phase,suffix in [('cuda_jl','shared','cuda.jl'),('ka','KA','KA'),('jacc','JACC','JACC')]:
  if phase not in phases:continue
  commit=phases[phase];main=R/f'src/{bench}-julia/{bench}-julia-{suffix}/main.jl';assert main.exists();assert subprocess.check_output(['git','show',f'{commit}:{main.relative_to(R)}'],cwd=R)==main.read_bytes(),('Implementation source changed',bench,m)
  child=main.parent;canonical=R/f'.porting-state/porting_logs/{bench}-julia-{m}/manifest.json';status=R/f'.porting-state/portable_julia/status_300/{bench}/{m}.json'
  for f,kind in [(canonical,'canonical'),(child/'translation_accounting.json','child_accounting'),(child/'study_manifest.json','child_manifest'),(status,'status')]:
   if not f.exists():continue
   d=json.loads(f.read_text());history=R/f'src/{bench}-julia/reference_history/accounting_original_delivery_20261020/{m}_{kind}_before_commit_join_{sha(f)[:12]}.json';history.parent.mkdir(parents=True,exist_ok=True)
   if not history.exists():history.write_bytes(f.read_bytes())
   d['source_commit']=commit;d['commit_status']='committed';d['coordinator_source_commit_join']={'commit':commit,'source':str(main.relative_to(R)),'source_sha256':sha(main),'phase':phase,'review':'git-show source bytes equal current verified main; metadata-only join, observed usage and raw GPU evidence preserved','original_metadata':str(history.relative_to(R)),'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   temp=f.with_suffix('.commitjoin.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(f)
  joined.append({'benchmark':bench,'model':m,'source_commit':commit,'main_sha256':sha(main)})
(P/'source_commit_receipts.json').write_text(json.dumps({'source_commits':receipts,'joins':joined,'scope':'Separate verified-source commits; metadata joining does not change outputs, acceptance or observed token counters.'},indent=2)+'\n');print(len(receipts),'benchmarks;',len(joined),'source-commit joins')
