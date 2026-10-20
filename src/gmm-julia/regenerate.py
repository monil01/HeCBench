"""Recover original DVC data only; no downloaded code or GPU execution."""
import argparse,hashlib,json,tarfile,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--asset-dir',type=Path);a=p.parse_args();B=Path(__file__).resolve().parent;R=B.parents[1];A=a.asset_dir or R/'.porting-state/portable_julia/external_assets/gmm';A.mkdir(parents=True,exist_ok=True);meta=json.loads((B/'asset_provenance.json').read_text());f=A/'data.tar.gz'
if not f.exists():
 with urllib.request.urlopen(meta['url'],timeout=60)as response:data=response.read(meta['bytes']+1)
 assert len(data)==meta['bytes'];f.write_bytes(data)
data=f.read_bytes();assert len(data)==meta['bytes']and hashlib.md5(data).hexdigest()==meta['archive_md5']and hashlib.sha256(data).hexdigest()==meta['archive_sha256']
with tarfile.open(f,'r:gz')as t:
 files=[m for m in t.getmembers()if m.isfile()];assert len(files)==len(meta['files'])
 for m in files:
  q=Path(m.name);assert not q.is_absolute()and '..'not in q.parts;v=t.extractfile(m).read();e=next(e for e in meta['files']if e['path']==m.name);assert len(v)==e['bytes']and hashlib.sha256(v).hexdigest()==e['sha256'];out=A/q;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(v)
print(json.dumps({'data_recovered':True,'asset_directory':str(A),'never_executed_downloaded_code':True}))
