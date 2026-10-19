"""Recover hash-pinned author tables and regenerate FRNA fixtures; no GPU calls.
Run from any directory. Never executes downloaded code; regular data files only.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile,tempfile,urllib.request
p=argparse.ArgumentParser();p.add_argument('--asset-dir',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--cxx',default='g++');a=p.parse_args();B=Path(__file__).resolve().parent;R=B.parents[1];A=a.asset_dir or R/'.porting-state/portable_julia/external_assets/frna';O=a.output_dir or B;A.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
provenance=json.loads((B/'asset_provenance.json').read_text())
for item in provenance['archives']:
 name=Path(item['declared_DVC']).name.removesuffix('.dvc');archive=A/name
 if not archive.exists():
  with urllib.request.urlopen(item['url'],timeout=60) as response:
   raw=response.read(item['bytes']+1)
  assert len(raw)==item['bytes'];archive.write_bytes(raw)
 raw=archive.read_bytes();assert len(raw)==item['bytes'] and hashlib.md5(raw).hexdigest()==item['md5'] and hashlib.sha256(raw).hexdigest()==item['sha256']
 with tarfile.open(archive,'r:gz')as t:
  regular=[m for m in t.getmembers()if m.isfile()];assert len(regular)==len(item['entries'])
  for m in regular:
   q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts
   matches=[e for e in item['entries']if e['path']==m.name];assert len(matches)==1;e=matches[0];data=t.extractfile(m).read();assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'];target=A/q;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
with tempfile.TemporaryDirectory(prefix='frna_fixture_')as tmp:
 exe=Path(tmp)/'generate';c=B/'configured_native'
 subprocess.run([a.cxx,'-O3','-DINT','-std=c++17','-I'+str(c),str(B/'generate_fixture.cpp'),str(c/'fparam.c'),str(c/'fbase.c'),str(c/'util.c'),'-lm','-o',str(exe)],check=True)
 subprocess.run([str(exe),str(A/'data_tables'),str(c/'RD0260.seq'),str(O)],check=True)
expected=json.loads((B/'regeneration_contract.json').read_text())
for name,want in expected['regenerated_files'].items():
 data=(O/name).read_bytes();assert len(data)==want['bytes'] and hashlib.sha256(data).hexdigest()==want['sha256'],name
print(json.dumps({'regeneration_pass':True,'output_dir':str(O),'assets':str(A),'never_executed_downloaded_code':True}))
