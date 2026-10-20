#!/usr/bin/env python3
"""Recover the exact small DVC-declared XLQC example, never substitute a molecule."""
from pathlib import Path
import argparse,json,urllib.request,hashlib,tarfile,io
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);m=json.load(open(P/'asset_provenance.json'));p=z.output/'example.tar.gz'
if p.exists():b=p.read_bytes()
else:
 with urllib.request.urlopen(m['url'],timeout=60)as r:assert r.status==200;b=r.read(m['archive_bytes']+1)
 assert len(b)==m['archive_bytes'];p.write_bytes(b)
assert len(b)==m['archive_bytes']and hashlib.md5(b).hexdigest()==m['archive_md5']and hashlib.sha256(b).hexdigest()==m['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz')as t:
 for x in m['assets']:
  member=t.getmember(x['name']);assert member.isfile()and member.size==x['bytes']and'..'not in Path(member.name).parts and not Path(member.name).is_absolute();data=t.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==x['sha256'];out=z.output/member.name;out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():assert out.read_bytes()==data
  else:out.write_bytes(data)
print('EXACT_XLQC_DVC_INPUTS_PASS')
