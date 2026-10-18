"""CPU-only isolated cold regeneration; compare hashes without overwriting measured assets."""
from pathlib import Path
import hashlib,json,subprocess,shutil,tempfile,sys
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[1]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
with tempfile.TemporaryDirectory(prefix='hecbench-bonds-cold-')as tmp:
 root=Path(tmp)/'repo';dest=root/'src/bonds-julia';dest.mkdir(parents=True);native=root/'src/bonds-cuda';native.mkdir();work=root/'.porting-state/portable_julia/bonds_reference_review';work.mkdir(parents=True)
 for name in ['source_cpu_witness.cpp','generate_source_inputs.cpp','independent_default_oracle.py','generate_fixtures.py']:shutil.copy2(B/name,dest/name)
 for name in ['bondsStructs.h','bondsKernelsCpu.cu','bondsKernelsCpu.h']:shutil.copy2(R/'src/bonds-cuda'/name,native/name)
 for source,out in [('source_cpu_witness.cpp','cpu_witness'),('generate_source_inputs.cpp','generate_source_inputs')]:subprocess.run(['g++','-O3','-std=c++17',str(dest/source),'-o',str(work/out)],check=True)
 with (work/'ABI.json').open('w')as f:subprocess.run([str(work/'cpu_witness')],stdout=f,check=True)
 with (work/'source_generation.json').open('w')as f:subprocess.run([str(work/'generate_source_inputs'),'1000000','256'],cwd=work,stdout=f,check=True)
 for name in ['independent_default_oracle.py','generate_fixtures.py']:subprocess.run([sys.executable,str(dest/name)],stdout=subprocess.DEVNULL,check=True)
 assets=[]
 for name in ['input_u32.bin','oracle_f64.bin','cases.json','study_cases.h','study_cases.jl']:
  old=B/name;new=dest/name;h=sha(old);cold=sha(new);assert h==cold,(name,h,cold);assets.append(dict(path=str(old.relative_to(R)),bytes=old.stat().st_size,measured_sha256=h,cold_sha256=cold,exact_reproduction=True))
 proof=dict(scope='CPU-only fresh temporary source tree, compile actual source initializer/CPUwitness, independentF64formula/defaultregime, fixtures+caseheaders regeneratehashidentically. Measuredbytesneveroverwritten, zeroGPU.',command='python3 src/bonds-julia/reproduce_large_fixtures.py',python=sys.version,numpy=np.__version__,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],assets=assets,source_recipes=[dict(path=str((B/n).relative_to(R)),sha256=sha(B/n))for n in ['source_cpu_witness.cpp','generate_source_inputs.cpp','independent_default_oracle.py','generate_fixtures.py']]);(B/'cold_generation_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(assets))
