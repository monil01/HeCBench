"""Actual child GPU program plus independent, untimed complete warmtrace/ROI audit."""
import subprocess,sys,os,shutil,json,hashlib
from pathlib import Path
p=Path('/home/imo/HeCBench_julia/src/snicit-julia')
r=subprocess.run(sys.argv[1:])
if r.returncode:sys.exit(r.returncode)
r=subprocess.run([sys.executable,str(p/'audit_model.py'),os.environ['STUDY_OUTPUT_DIR']])
if r.returncode:sys.exit(r.returncode)
d=Path(os.environ['STUDY_OUTPUT_DIR']);dest=d/'retained_diagnostics';dest.mkdir(parents=True,exist_ok=True)
for f in [*d.glob('trace_case*.bin'),d/'independent_cpu_audit.json']:shutil.copy2(f,dest/f.name)
print('SNICIT_RETAINED_TRACES',str(dest),flush=True)
