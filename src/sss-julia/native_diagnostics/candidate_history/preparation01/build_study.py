#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];G=R/'.porting-state/portable_julia/sss_readiness/gsl-prefix';B=Path(sys.argv[1]).resolve();B.mkdir(parents=True,exist_ok=True)
commands=[['g++','-std=c++17','-O3','-fPIC','-shared','-I/usr/include','-I'+str(G/'include'),str(P/'study_runtime.cpp'),str(G/'lib/libgsl.a'),str(G/'lib/libgslcblas.a'),'-lcudart','-lm','-o',str(B/'libsss_host.so')],['/usr/bin/nvcc','-std=c++17','-O3','-arch=sm_70','-allow-unsupported-compiler','-cudart','shared',str(P/'study_native.cu'),'-L'+str(B),'-lsss_host','-Xlinker','-rpath','-Xlinker',str(B),'-o',str(B/'sss_native')]]
for i,c in enumerate(commands):
 p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(B/f'build_{i}.log').write_text(p.stdout);print('build',i,'exit',p.returncode);print(p.stdout[-2500:]);
 if p.returncode:raise SystemExit(p.returncode)
(B/'build_receipt.json').write_text(json.dumps({'commands':commands,'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in B.iterdir()if f.is_file()}},indent=2)+'\n')
