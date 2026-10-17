#!/usr/bin/env python3
"""Build the source-formula fixture and independent Double Gram-LS oracle."""
from pathlib import Path
import tempfile,subprocess,json,hashlib,argparse
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--create',action='store_true');a=ap.parse_args()
with tempfile.TemporaryDirectory(prefix='hecbench-lomb-fixture-')as td:
 t=Path(td);exe=t/'generator';subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off',str(P/'generate_study_fixture.cpp'),'-o',str(exe)],check=True);inputs=t/'input.bin';expected=t/'expected.bin';r=subprocess.run([str(exe),str(inputs),str(expected)],check=True,text=True,capture_output=True);meta=json.loads(r.stdout);meta['files']={}
 for source,name in [(inputs,'study_input_f32.bin'),(expected,'study_expected_f64.bin')]:
  data=source.read_bytes();dest=P/name
  if a.create and not dest.exists():dest.write_bytes(data)
  else:assert dest.read_bytes()==data
  meta['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 meta['compiler_flags']=['g++','-O3','-std=c++17','-ffp-contract=off'];meta['generator_source_sha256']=hashlib.sha256((P/'generate_study_fixture.cpp').read_bytes()).hexdigest();meta['libm_reproducibility_scope']='UseslocalC++host sinf/sin/cos, exacthashcheck; differentlibm/toolchainmaychangebytes. Sharedsavedbinaryisimmutablemeasuredfixture.';print(json.dumps(meta,indent=2))
