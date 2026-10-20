from pathlib import Path
import argparse,subprocess,json,hashlib
S=Path(__file__).resolve().parent;O=S.parents[1]/'tsne-cuda';p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();sources=[S/'study_native.cu',*sorted(f for f in O.glob('*.cu')if f.name not in['main.cu','fit_tsne.cu'])];cmd=['/usr/bin/nvcc','-O3','--use_fast_math','-std=c++17','-arch=sm_70','-allow-unsupported-compiler','-include',str(S/'guard_allocator.h'),'-I'+str(O),*[str(f)for f in sources],'-o',str(a.output.resolve())];print(cmd,flush=True);subprocess.run(cmd,check=True)
