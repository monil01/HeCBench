"""CPU-only cold regeneration in a temporary tree; never overwrite measured inputs."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile,sys
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[1]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
with tempfile.TemporaryDirectory(prefix='hecbench-su3-cold-')as tmp:
 root=Path(tmp)/'repo';dest=root/'src/su3-julia';dest.mkdir(parents=True);native=root/'src/su3-cuda';native.mkdir();work=root/'.porting-state/portable_julia/su3_reference_review';work.mkdir(parents=True)
 for name in ['source_initializer_witness.cpp','generate_fixtures.py']:shutil.copy2(B/name,dest/name)
 for name in ['lattice.hpp','su3.h']:shutil.copy2(R/'src/su3-cuda'/name,native/name)
 subprocess.run(['g++','-O3','-std=c++17',str(dest/'source_initializer_witness.cpp'),'-o',str(work/'witness')],check=True)
 with(work/'initializer_layout.json').open('w')as f:subprocess.run([str(work/'witness')],cwd=work,stdout=f,check=True)
 result=subprocess.run([sys.executable,str(dest/'generate_fixtures.py')],capture_output=True,text=True,check=True)
 proof=[]
 for name in ['input_u32.bin','oracle_u32.bin','study_cases.h','study_cases.jl','cases.json']:
  measured=B/name;generated=dest/name;original_sha=sha(measured);new_sha=sha(generated);assert original_sha==new_sha,(name,original_sha,new_sha);proof.append(dict(path=str(measured.relative_to(R)),bytes=measured.stat().st_size,measured_sha256=original_sha,cold_regeneration_sha256=new_sha,exact_reproduction=True))
 report=dict(scope='CPU-only fresh source initializer compilation and deterministic full large-fixture regeneration in isolated temporary tree; measured input/oracle bytes never overwritten, zero GPU execution.',command='PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/su3-julia/reproduce_large_fixtures.py',python_version=sys.version,numpy_version=np.__version__,compiler_version=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],generator_sha256=sha(B/'generate_fixtures.py'),initializer_source_sha256=sha(B/'source_initializer_witness.cpp'),CPU_layout=json.loads((work/'initializer_layout.json').read_text()),assets=proof)
 (B/'cold_generation_proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(proof))
