from pathlib import Path
import tempfile,shutil,subprocess,sys,json,hashlib,platform,numpy as np
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='sps2d-assets-')as td:
 d=Path(td);shutil.copy2(P/'generate_fixtures.py',d/'generate_fixtures.py');r=subprocess.run([sys.executable,str(d/'generate_fixtures.py')],capture_output=True,text=True);assert not r.returncode,(r.stdout,r.stderr);rows=[]
 for name in ['input_u32.bin','oracle_u32.bin','study_plan.json','study_cases.h','study_cases.jl']:
  h=sha(d/name);assert h==sha(P/name);rows.append(dict(path=name,bytes=(d/name).stat().st_size,sha256=h,byteexact=True))
 (P/'generation_reproduction.json').write_text(json.dumps(dict(correct=True,assets=rows,CPU_only=True,GPU_executions=0,measured_assets_written=False,Python=platform.python_version(),NumPy=np.__version__,stdout=r.stdout),indent=2)+'\n');print('SPS2D_CPU_REPRODUCTION',len(rows),'byteexact',flush=True)
