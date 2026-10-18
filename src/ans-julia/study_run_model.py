"""Retain every actual scratch/warmtrace capture; independent CPU audit before acceptance."""
from pathlib import Path
import subprocess,sys,os,shutil
P=Path(__file__).resolve().parent;D=Path(os.environ['STUDY_OUTPUT_DIR']);dest=P.parents[1]/'.porting-state/portable_julia/diagnostics/ans'/D.name;dest.mkdir(parents=True,exist_ok=True)
r=subprocess.run(sys.argv[1:]);status=r.returncode
# Preserve diagnostic state even if a kernel/API/full-byte check failed.
for f in [*D.glob('warm_case*.bin'),*D.glob('trial*_case*_final.bin')]:shutil.copy2(f,dest/f.name)
print('ANS_RETAINED_STATES',str(dest),flush=True)
if status:sys.exit(status)
r=subprocess.run([sys.executable,str(P/'audit_model.py'),str(D),str(dest)])
if (D/'independent_cpu_audit.json').exists():shutil.copy2(D/'independent_cpu_audit.json',dest/'independent_cpu_audit.json')
sys.exit(r.returncode)
