#!/usr/bin/env python3
import pathlib,os,subprocess,json,argparse,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[2]
a=argparse.ArgumentParser();a.add_argument('--config',type=pathlib.Path,required=True);z=a.parse_args();cfg=z.config.resolve();assert cfg.is_relative_to(P)
env=os.environ.copy();env['STATE']='/tmp/hecbench-stencil-ports-state';env['PYTHONPATH']='/tmp/pasc-julia-plot-tools';env['OPENBLAS_NUM_THREADS']='1';tool=R/'tools/verify_portable_julia.py';r=subprocess.run(['python3',str(tool),'word2vec','--config',str(cfg)],env=env,cwd=R,text=True,capture_output=True);print(r.stdout,end='');print(r.stderr,end='')
for line in r.stdout.splitlines():
 try:item=json.loads(line)
 except ValueError:continue
 if 'result'not in item:continue
 real=pathlib.Path(item['result']);alias=R/'.porting-state/portable_julia/runs'/real.parent.name
 if not alias.exists():alias.symlink_to(real.parent,target_is_directory=True)
 else:assert alias.resolve()==real.parent.resolve()
 d={'actual':str(real),'canonical':str(alias/'result.json'),'sha256':hashlib.sha256(real.read_bytes()).hexdigest(),'storage':'Prospectiveowned/tmp, noolddatamoves/deletion; ephemeraluntilpublicresearcharchival','config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest()};out=P/'storage_provenance';out.mkdir(exist_ok=True);(out/(real.parent.name+'.json')).write_text(json.dumps(d,indent=2)+'\n')
raise SystemExit(r.returncode)
