#!/usr/bin/env python3
from pathlib import Path
import urllib.request,hashlib,tarfile,json,sys
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
assets=[('HIV1-NL43.tar.gz','670d08f7244957c65b53fba03a11d82e',4910),('data_tables.tar.gz','6f4e61c747d87f9db9ede78c3ffb74a2',1574971),('test.tar.gz','21d5cf47723f273907587bf1d69a4d11',290)];records=[]
for name,md5,size in assets:
 p=out/name;url=f'https://hecbench-geekdude.s3.amazonaws.com/files/md5/{md5[:2]}/{md5[2:]}'
 if not p.exists():
  with urllib.request.urlopen(url,timeout=60) as response:
   b=response.read(size+1)
  assert len(b)==size,(name,len(b),size);assert hashlib.md5(b).hexdigest()==md5;p.write_bytes(b)
 b=p.read_bytes();assert len(b)==size and hashlib.md5(b).hexdigest()==md5
 with tarfile.open(p) as t:
  members=t.getmembers()
  for m in members:
   dest=(out/m.name).resolve();assert dest.is_relative_to(out.resolve()) and not(m.issym() or m.islnk())
  t.extractall(out,filter='data')
 records.append({'archive':name,'source_url':url,'bytes':size,'md5':md5,'sha256':hashlib.sha256(b).hexdigest(),'members':[{'name':m.name,'bytes':m.size} for m in members]})
(out/'asset_provenance.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
