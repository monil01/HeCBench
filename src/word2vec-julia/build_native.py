#!/usr/bin/env python3
"""Fresh source-observer build only; no GPU, originalCUDA unchanged."""
import pathlib,argparse,subprocess,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--nvcc',default='/usr/bin/nvcc');z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);binary=z.output/'study_native';cmd=[z.nvcc,'-O3','-std=c++17','-arch=sm_70','-allow-unsupported-compiler',str(P/'study_native.cu'),'-o',str(binary)]
with open(z.output/'build.log','w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
d=dict(status='native_CPU_build_success',command=cmd,binary=str(binary),binary_sha256=sha(binary),build_log_sha256=sha(z.output/'build.log'),sources={str(p):sha(p)for p in [P/'study_native.cu',R/'src/word2vec-cuda/cbow.cu',R/'src/word2vec-cuda/cbow.h']},original_GPU_source_changed=False,GPU_execution=False);(z.output/'build_proof.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
