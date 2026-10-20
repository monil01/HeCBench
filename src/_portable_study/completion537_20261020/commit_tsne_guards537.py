from pathlib import Path
import subprocess,json,sys
R=Path('/home/imo/HeCBench_julia');rp='.porting-state/portable_julia/runs/20261020T151316499476Z_tsne/result.json';out=[]
for phase in ['shared','KA','JACC']:
 f=Path(f'/tmp/tsne_guard_commit_{phase}.log')
 with f.open('w')as log:v=subprocess.run([sys.executable,'tools/commit_verified_portable_variant.py','--benchmark','tsne','--result',rp,'--measurement-repair',*(['--shared']if phase=='shared'else['--variant',phase])],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 x={'phase':phase,'returncode':v.returncode,'log':str(f)}
 if v.returncode:x['error_tail']=f.read_text()[-1800:]
 else:x['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
 out.append(x);(R/'src/_portable_study/completion537_20261020/tsne_guard_commits.json').write_text(json.dumps({'result':rp,'phase':'additional guard measurement, originaltranslation counters unchanged','commits':out},indent=2)+'\n');print(x,flush=True)
 if v.returncode:break
