"""CPU-only rebuild sourcepreparer + scalaroracle and regenerate in isolated temp."""
from pathlib import Path
import tempfile,subprocess,shutil,sys,json,hashlib,platform
P=Path(__file__).resolve().parent;R=P.parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='ans-assets-')as td:
 d=Path(td);files=['ans_encoder.cc','ans_table_generator.cc','multians_codetable.cc','multians_input_buffer.cc'];cmd=['g++','-O3','-std=c++17','-I'+str(R/'src/ans-cuda/include'),str(P/'generate_source_fixtures.cpp')]+[str(R/'src/ans-cuda/src'/n)for n in files]+['-o',str(d/'generate')];build=subprocess.run(cmd,capture_output=True,text=True);assert not build.returncode,build.stderr;r=subprocess.run([str(d/'generate'),str(d)],capture_output=True,text=True);assert not r.returncode,(r.stdout,r.stderr);shutil.copy2(P/'prepare_plan.py',d/'prepare_plan.py');p=subprocess.run([sys.executable,str(d/'prepare_plan.py')],capture_output=True,text=True);assert not p.returncode,(p.stdout,p.stderr);rows=[]
 for n in ['input_u32.bin','oracle_u8.bin','source_cases.tsv','study_plan.json','study_cases.h']:
  h=sha(d/n);assert h==sha(P/n),n;rows.append(dict(path=n,bytes=(d/n).stat().st_size,sha256=h,byteexact=True))
 report=dict(correct=True,assets=rows,CPU_only=True,GPU_executions=0,measured_assets_written=False,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],platform=platform.platform(),Python=platform.python_version(),serial_roundtrip_source_cases=15,serial_roundtrip_bytes=15728640,stdout=r.stdout,original_CPU_source_files=[dict(path='src/ans-cuda/src/'+n,sha256=sha(R/'src/ans-cuda/src'/n))for n in files],source_generator_sha256=sha(P/'generate_source_fixtures.cpp'))
 (P/'generation_reproduction.json').write_text(json.dumps(report,indent=2)+'\n');print('ANS_CPU_REPRODUCTION',len(rows),'byteexact',flush=True)
