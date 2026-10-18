"""Actual child GPU program plus independent, untimed complete warmtrace/ROI audit."""
import subprocess,sys,os
from pathlib import Path
p=Path(__file__).resolve().parent
r=subprocess.run(sys.argv[1:])
if r.returncode:sys.exit(r.returncode)
r=subprocess.run([sys.executable,str(p/'audit_model.py'),os.environ['STUDY_OUTPUT_DIR']])
sys.exit(r.returncode)
