from pathlib import Path
import json,subprocess,sys,datetime
R=Path('/home/imo/HeCBench_julia');S=R/'.porting-state/portable_julia';c=json.loads(Path('/tmp/current537_census.json').read_text());new=set(json.loads((S/'batch_537.json').read_text())['new_benchmarks']);plan=json.loads(Path('/tmp/commit537_plan.json').read_text());by={x['benchmark']:x for x in plan}
for x in c['rows']:
 n=x['benchmark'];main=R/f'src/{n}-julia/{n}-julia-KA/main.jl'
 if n not in new or n in by or not main.exists():continue
 if subprocess.run(['git','ls-files','--error-unmatch',str(main.relative_to(R))],cwd=R,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:by[n]={'benchmark':n,'results':[x['result']]}
Path('/tmp/commit537_plan.json').write_text(json.dumps(list(by.values()),indent=2)+'\n');out=[]
for n,x in by.items():
 args=sum((['--result',p] for p in x['results']),[])
 entry={'benchmark':n,'results':x['results'],'phases':[]};out.append(entry)
 for phase in ['shared','KA','JACC']:
  logfile=Path(f'/tmp/commit537_{n}_{phase}.log');cmd=[sys.executable,'tools/commit_verified_portable_variant.py','--benchmark',n,*args,*(['--shared'] if phase=='shared' else ['--variant',phase])]
  with logfile.open('w') as f:v=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  e={'phase':phase,'returncode':v.returncode,'log':str(logfile)}
  if v.returncode==0:e['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
  else:e['error_tail']=logfile.read_text()[-1200:]
  entry['phases'].append(e);Path('/tmp/commit537_outcomes.json').write_text(json.dumps(out,indent=2)+'\n');print(n,phase,v.returncode,e.get('commit',e.get('error_tail')),flush=True)
  if v.returncode:break
  assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R,text=True).strip(),'Unexpected pending index'
print('Finished coordinator sourcecommit pass',len(out),flush=True)
