"""Cold CPU regeneration in isolated temp with unchanged original source data."""
from pathlib import Path
import tempfile,shutil,subprocess,sys,json,hashlib,platform,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='snicit-assets-')as td:
 root=Path(td);d=root/'src/snicit-julia';d.mkdir(parents=True);(root/'src/snicit-cuda').symlink_to(R/'src/snicit-cuda',target_is_directory=True);shutil.copy2(P/'generate_fixtures.py',d/'generate_fixtures.py');r=subprocess.run([sys.executable,str(d/'generate_fixtures.py')],capture_output=True,text=True);assert not r.returncode,(r.stdout,r.stderr)
 rows=[]
 for n in ['input_u32.bin','dense_oracle_f64.bin','study_plan.json','study_cases.h','study_cases.jl']:
  actual=sha(d/n);expected=sha(P/n);rows.append(dict(path=n,bytes=(d/n).stat().st_size,sha256=actual,expected_sha256=expected,byteexact=actual==expected));assert actual==expected,n
 (P/'generation_reproduction.json').write_text(json.dumps(dict(correct=True,assets=rows,CPU_only=True,GPU_executions=0,measured_assets_written=False,Python=platform.python_version(),NumPy=np.__version__,stdout=r.stdout),indent=2)+'\n');print('SNICIT_CPU_REPRODUCTION',len(rows),'byteexact',flush=True)
