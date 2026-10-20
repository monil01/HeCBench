#!/usr/bin/env python3
from pathlib import Path
import urllib.request,hashlib,tarfile,json,sys
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);md5='8cd52d0395bc0f76df02e438a4fa673d';size=1251585;p=out/'genome.tar.bz';url=f'https://hecbench-geekdude.s3.amazonaws.com/files/md5/{md5[:2]}/{md5[2:]}'
if not p.exists():
 with urllib.request.urlopen(url,timeout=60)as r:b=r.read(size+1)
 assert len(b)==size and hashlib.md5(b).hexdigest()==md5;p.write_bytes(b)
b=p.read_bytes();assert len(b)==size and hashlib.md5(b).hexdigest()==md5
with tarfile.open(p)as t:
 members=t.getmembers()
 for m in members:assert(out/m.name).resolve().is_relative_to(out.resolve())and not(m.issym()or m.islnk())
 t.extractall(out,filter='data')
r={'url':url,'archive_bytes':size,'md5':md5,'sha256':hashlib.sha256(b).hexdigest(),'members':[{'name':m.name,'bytes':m.size}for m in members]};(out/'asset_provenance.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
