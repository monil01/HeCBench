#!/usr/bin/env python3
"""Regenerate declared input/oracle fixtures with CPU-only compilation, no GPU.
Use in a separate output directory; committed inputs are never silently overwritten.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
p=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);ap.add_argument('--cxx',default='g++');a=ap.parse_args();out=Path(a.output_dir).resolve();assert out!=p,'Use a separate directory to preserve measured input assets';out.mkdir(parents=True,exist_ok=True);cxx=shutil.which(a.cxx);assert cxx,'C++ compiler unavailable; no system package installation attempted'
names=['diffusive','localized','small_lattice','general_tail','spin_pairs'];sources=[p/'study_fixture_builder.cpp']+[p/'original_native'/n for n in ['lsqt.cu','sigma.cu','vector.cu','hamiltonian.cu','model.cu','model_general.cu','model_lattice.cu','anderson.cu','charge.cu']];commands=[]
def run(argv):
 commands.append([str(x)for x in argv]);subprocess.run(commands[-1],check=True)
run([cxx,'-O3','-std=c++17','-DDEBUG','-DCPU_ONLY','-x','c++',*sources,'-o',out/'study_fixture_builder'])
for name in names:run([out/'study_fixture_builder',p/'cases'/name,out/(name+'_fixture.bin')])
run([cxx,'-O3','-std=c++17',p/'study_math_oracle.cpp','-o',out/'study_math_oracle']);run([out/'study_math_oracle',out,out/'independent_math_oracle.bin'])
rows=[]
for f in sorted(out.glob('*.bin')):
 digest=hashlib.sha256(f.read_bytes()).hexdigest();original=p/f.name;expected=hashlib.sha256(original.read_bytes()).hexdigest()if original.exists()else None;rows.append({'path':str(f),'sha256':digest,'recorded_sha256':expected,'byte_identical_recorded':digest==expected})
(out/'regeneration_receipt.json').write_text(json.dumps({'commands':commands,'assets':rows,'scope':'CPU source-definedinput/oraclepreparation only; not GPUverification. stdlib/compiler/oracleprecision differences may affect reproducibility; mismatches remain explicit.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
