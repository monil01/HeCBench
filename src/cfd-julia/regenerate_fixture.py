"""Regenerate the full097K CFD fixture and independent Float64 oracle.
External public Rodinia mesh is hash-checked; large data stays outside Git.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--cxx',default='g++');p.add_argument('--threads',type=int,default=8);a=p.parse_args();root=Path(__file__).resolve().parent
expected='43534e58454ba8baf95253e14d3b07150b9960577e922a954d6e51142a27abfa'
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(a.mesh)==expected,'Wrong public097K mesh bytes'
f32line='0.166666672 -1 -0.5 0 0 -1 0 -0.5 0 -1 0 0 -0.5 -1 0.5 0.5 0.5'
(root/'freestream192.mesh').write_text('192\n'+(f32line+'\n')*192)
# The line encoding matches the first source-generation script exactly.
with tempfile.TemporaryDirectory(prefix='cfd-cpu-oracle-')as tmp:
 exe=Path(tmp)/'oracle';subprocess.run([a.cxx,'-O3','-fopenmp','-std=c++17',str(root/'generate_fixture.cpp'),'-o',str(exe)],check=True)
 subprocess.run([str(exe),str(a.mesh.resolve())],cwd=root,env={**os.environ,'OMP_NUM_THREADS':str(a.threads)},check=True)
for name in ['freestream192.mesh','fixture.bin','oracle.bin']:
 print(name,(root/name).stat().st_size,sha(root/name))
