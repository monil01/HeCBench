from pathlib import Path
import json,subprocess,hashlib,os
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;out=R/'.porting-state/portable_julia/builds/f16sp';out.mkdir(parents=True,exist_ok=True)
q=subprocess.run(['nvidia-smi','-i',os.getenv('CUDA_VISIBLE_DEVICES','6'),'--query-gpu=compute_cap','--format=csv,noheader'],text=True,capture_output=True,check=True);arch='sm_'+q.stdout.strip().replace('.','')
command=['nvcc','-O3','-std=c++17','-arch='+arch,'--shared','-Xcompiler','-fPIC',str(p/'vendor_bridge.cu'),'-o',str(out/'libstudy_vendor.so'),'-lcublas','-ldl']
r=subprocess.run(command,text=True,capture_output=True);(out/'build.log').write_text(r.stdout+r.stderr)
version=subprocess.run(['nvcc','--version'],text=True,capture_output=True).stdout
record=dict(command=command,returncode=r.returncode,nvcc=version,architecture=arch,log=str((out/'build.log').relative_to(R)),source_hashes={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [p/'vendor_bridge.cu',R/'src/f16sp-cuda/main.cu',R/'src/f16sp-cuda/kernels.h']},binary_sha256=hashlib.sha256((out/'libstudy_vendor.so').read_bytes()).hexdigest()if r.returncode==0 else None)
(p/'vendor_build_provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(r.returncode)
