from pathlib import Path
import json,subprocess,sys
R=Path('/home/imo/HeCBench_julia');out=[]
for x in json.loads(Path('/tmp/retry_combined537_plan.json').read_text()):
 n=x['benchmark'];entry={'benchmark':n,'results':x['results'],'phases':[]};out.append(entry);args=sum((['--result',p]for p in x['results']),[])
 for phase in ['shared','KA','JACC']:
  log=Path(f'/tmp/combined537_{n}_{phase}.log');cmd=[sys.executable,'tools/commit_verified_portable_variant.py','--benchmark',n,*args,'--combined-reference-review',f'src/{n}-julia/combined_qualification_review.json',*(['--shared']if phase=='shared'else['--variant',phase])]
  with log.open('w')as f:v=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  e={'phase':phase,'returncode':v.returncode,'log':str(log)}
  if v.returncode:e['error_tail']=log.read_text()[-2000:]
  else:e['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
  entry['phases'].append(e);Path('/tmp/combined537_outcomes.json').write_text(json.dumps(out,indent=2)+'\n');print(n,phase,e,flush=True)
  if v.returncode:break
