"""CPU-only deterministic fixture/full all-pairs oracle regeneration."""
from pathlib import Path
import argparse,subprocess,tempfile,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);p.add_argument('--cxx',default='g++');a=p.parse_args();B=Path(__file__).resolve().parent;O=a.output_dir or B;O.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='halo_fixture_')as temp:
 exe=Path(temp)/'generate';subprocess.run([a.cxx,'-O3','-std=c++17','-fopenmp',str(B/'generate_fixture.cpp'),'-o',str(exe)],check=True);subprocess.run([str(exe),str(O.resolve())],check=True)
x=json.loads((B/'regeneration_contract.json').read_text())
for name,want in x['files'].items():
 raw=(O/name).read_bytes();assert len(raw)==want['bytes']and hashlib.sha256(raw).hexdigest()==want['sha256'],name
print(json.dumps({'all_hashes_pass':True,'output_directory':str(O),'GPU_execution':False,'source_original_N2_fulltree_support_claim':False}))
