
from pathlib import Path
import subprocess,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'.porting-state/portable_julia/runtime/mallocFree';O.mkdir(parents=True,exist_ok=True)
cmd=['g++','-O3','-std=c++17','-shared','-fPIC',str(P/'study_runtime.cpp'),'-o',str(O/'libstudy_runtime.so'),'-ldl']
x=subprocess.run(cmd,text=True,capture_output=True);(O/'build.log').write_text(x.stdout+x.stderr)
record=dict(command=cmd,returncode=x.returncode,compiler=subprocess.run(['g++','--version'],text=True,capture_output=True).stdout,source_sha256=hashlib.sha256((P/'study_runtime.cpp').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((O/'libstudy_runtime.so').read_bytes()).hexdigest()if x.returncode==0 else None,log=str((O/'build.log').relative_to(R)),device_compiler_invoked=False,device_kernels=0)
(P/'runtime_build_provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(x.returncode)
