#!/usr/bin/env python3
"""Rebuild all CPU data/oracles from deterministic recipes, without GPU work."""
from pathlib import Path
import argparse,subprocess,tempfile,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();P=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);fixtures=json.loads((P/'fixture_manifest.json').read_text());oracles=json.loads((P/'oracle_manifest.json').read_text());expected={c['file']:c['sha256']for c in fixtures['cases']};expected.update({c['path']:c['sha256']for c in oracles['cases']})
for name in ['fixture_manifest.json','oracle_manifest.json']:expected[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='si-cold-cpu-')as tmp:
 tmp=Path(tmp);subprocess.run(['/usr/bin/g++','-O3','-std=c++17',str(P/'generate_original_uniform.cpp'),'-o',str(tmp/'gen')],check=True);subprocess.run([str(tmp/'gen'),str(tmp/'original.bin')],check=True);subprocess.run([sys.executable,str(P/'prepare_control_assets.py'),str(out),'--original-data',str(tmp/'original.bin')],check=True);subprocess.run([sys.executable,str(P/'prepare_oracles.py'),str(out)],check=True)
records=[]
for name,sha in expected.items():
 f=out/name;actual=hashlib.sha256(f.read_bytes()).hexdigest();assert actual==sha,(name,actual,sha);records.append({'file':name,'bytes':f.stat().st_size,'sha256':actual,'matches_frozen_CPU_asset':True})
compiler=subprocess.run(['/usr/bin/g++','--version'],check=True,capture_output=True,text=True).stdout
(out/'regeneration_provenance.json').write_text(json.dumps({'status':'pass','CPU_only':True,'Python':sys.version,'g++':compiler,'all_files':records,'oracle':'IndependentfullPythonsetintersections; noGPUoutputsused','CXX_distribution_scope':'GCC13.3/libstdc++ originaluniformcasebytesvalidatedagainstactualoriginalgenerator; differentSTLimplementations mayfailhashcheck'},indent=2)+'\n');print('ALL_CPU_FILES_BYTE_EXACT',len(records))
