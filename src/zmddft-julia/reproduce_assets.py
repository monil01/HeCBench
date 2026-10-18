"""Regenerate deterministic CPU assets and kernel sources in isolation; no GPU."""
from pathlib import Path
import tempfile,shutil,subprocess,hashlib,json,sys
P=Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for a in iter(lambda:f.read(4*1024*1024),b''):h.update(a)
 return h.hexdigest()
checks=[]
with tempfile.TemporaryDirectory(prefix='zmddft-cold-assets-')as tmp:
 T=Path(tmp);Q=T/'src/zmddft-julia';C=T/'src/zmddft-cuda';Q.mkdir(parents=True);C.mkdir(parents=True);(Q/'zmddft-julia-cuda.jl').mkdir();shutil.copy2(R/'src/zmddft-cuda/main.cu',C/'main.cu')
 for n in ['generate_kernels.py','generate_fixtures.py']:shutil.copy2(P/n,Q/n)
 for model in ['cuda_jl','ka','jacc']:subprocess.run([sys.executable,str(Q/'generate_kernels.py'),'--model',model],check=True,stdout=subprocess.PIPE)
 subprocess.run([sys.executable,str(Q/'generate_fixtures.py')],check=True,stdout=subprocess.PIPE)
 names=['input_complex_f64.bin','twiddles_f64.bin','oracle_stage0_f64.bin','oracle_stage1_f64.bin','oracle_stage2_f64.bin','impulse_stage0_f64.bin','impulse_stage1_f64.bin','impulse_stage2_f64.bin','study_plan.json','zmddft-julia-cuda.jl/main.jl','zmddft-julia-KA/main.jl','zmddft-julia-JACC/main.jl']
 for name in names:
  a=sha(P/name);b=sha(Q/name);assert a==b,(name,a,b);checks.append(dict(path=str((P/name).relative_to(R)),sha256=a,bytes=(P/name).stat().st_size))
(P/'generation_reproduction.json').write_text(json.dumps(dict(CPU_only=True,GPU_runs=0,existing_measured_files_written=False,Python=sys.version,method='Temporarysrcstructure/originalCsourcecopy, all3kernelgeneratormodes and fullCPUfixture/oraclegeneration; everybytehashcompared.',checks=checks),indent=2)+'\n');print('Coldassetsbyteexact',len(checks),'files')
