"""Regenerate all CPU source fixtures and serial witness in an isolated tree; no GPU use."""
from pathlib import Path
import hashlib,json,subprocess,shutil,tempfile,sys
B=Path(__file__).resolve().parent;R=B.parents[1]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
names=[f'{s}{i}.bin'for s in ['case','params','regions']for i in range(7)]+['source_cpu_oracle_f64.bin']
recipes=['generate_source_inputs.cpp','study_setup.h','source_volume.cpp','source_cpu_math.h','study_native.cu']
with tempfile.TemporaryDirectory(prefix='hecbench-lulesh-cold-')as temp:
 root=Path(temp)/'repo';dest=root/'src/lulesh-julia';native=root/'src/lulesh-cuda';dest.mkdir(parents=True);native.mkdir()
 for n in recipes:shutil.copy2(B/n,dest/n)
 for n in ['lulesh.h','lulesh-init.cu']:shutil.copy2(R/'src/lulesh-cuda'/n,native/n)
 common=['g++','-O3','-std=c++17','-D__host__=','-D__device__=','-I/usr/local/cuda-12.0/include','-x','c++']
 subprocess.run(common+[str(dest/'generate_source_inputs.cpp'),str(dest/'source_volume.cpp'),str(native/'lulesh-init.cu'),'-o',str(dest/'generate')],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
 subprocess.run([str(dest/'generate')],cwd=dest,stdout=subprocess.DEVNULL,check=True)
 subprocess.run(common+['-ffp-contract=off','-DCPU_REFERENCE',str(dest/'study_native.cu'),str(native/'lulesh-init.cu'),'-o',str(dest/'pipeline')],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
 subprocess.run([str(dest/'pipeline')],cwd=dest,stdout=subprocess.DEVNULL,check=True)
 assets=[]
 for n in names:
  measured=sha(B/n);cold=sha(dest/n);assert measured==cold,n
  assets.append(dict(path=str((B/n).relative_to(R)),bytes=(B/n).stat().st_size,measured_sha256=measured,cold_sha256=cold,exact_reproduction=True))
 proof=dict(scope='Fresh temporary source tree; actual original Domain constructor/regions and serial unchanged fifteen numerical bodies regenerate22 CPU assets bit-identically. Source CPU witness shares source equations, separate from independent geometric/physical audit. No GPU calls, measured assets untouched.',command='python3 src/lulesh-julia/reproduce_large_fixtures.py',python=sys.version,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],CPU_floating_policy='O3, ffp-contract=off for full serial witness',recipes=[dict(path=str((B/n).relative_to(R)),sha256=sha(B/n))for n in recipes],assets=assets)
 (B/'cold_generation_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('CPU cold regeneration PASS',len(assets),'hash-identical assets')
