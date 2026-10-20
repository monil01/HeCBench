#!/usr/bin/env python3
"""Unprivileged fresh regeneration, refuses existing output; GPU6 serialized."""
import argparse,pathlib,subprocess,tempfile,os,shutil,json,hashlib
P=pathlib.Path(__file__).resolve().parent
q=argparse.ArgumentParser();q.add_argument('--output',type=pathlib.Path,required=True);a=q.parse_args();out=a.output.resolve()
if out.exists():raise SystemExit('Refusing to overwrite any existing fixture/evidence directory: '+str(out))
base=out.parent
if not base.exists():raise SystemExit('Create intended output parent explicitly before regeneration')
if shutil.disk_usage(base).free<2*1024**3:raise SystemExit('Need2GiB free reserve before bounded~150MiB fixture generation')
out.mkdir();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='6';commands=[]
with tempfile.TemporaryDirectory(prefix='hecbench-rbc-build-')as t:
 binary=pathlib.Path(t)/'prepare'
 commands.append(['flock','-x','/tmp/hecbench-julia-study-gpu.lock','/usr/bin/nvcc','-std=c++17','-O3','-arch=sm_70',str(P/'full_study.cu'),'-lcublas','-o',str(binary)])
 commands.append(['flock','-x','/tmp/hecbench-julia-study-gpu.lock',str(binary),str(out),'1'])
 for index,c in enumerate(commands):
  with (out/('native_prepare_'+str(index)+'.log')).open('w')as f:subprocess.run(c,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (out/'independent_oracle.log').open('w')as f:subprocess.run(['julia','--startup-file=no',str(P/'independent_oracle.jl'),str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
assert json.load(open(out/'independent_oracle_report.json'))['pass']
record={'fixture_directory':str(out),'CPU_oracle':'FullFloat64source-shaped1000stepRK3; realadaptivehistory independentlychecked; code/engineeringlimits inverification_plan.md','files':[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in out.iterdir()if p.is_file()]}
(out/'regeneration_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(out)
