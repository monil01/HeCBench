#!/usr/bin/env python3
"""Retrieve the exact declared SS input object with bounded size and digest gates."""
import argparse,json,pathlib,urllib.request,hashlib,tarfile,io
P=pathlib.Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);m=json.load(open(P/'asset_provenance.json'));expected=m['archive'];target=z.output/'input.tar.gz'
if target.exists():b=target.read_bytes()
else:
 with urllib.request.urlopen(m['url'],timeout=60)as r:
  assert r.status==200;b=r.read(expected['bytes']+1)
 assert len(b)==expected['bytes'];target.write_bytes(b)
assert len(b)==expected['bytes']and hashlib.md5(b).hexdigest()==expected['md5']and hashlib.sha256(b).hexdigest()==expected['sha256']
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz')as t:
 for x in m['assets']:
  f=t.getmember(x['name']);assert f.isfile()and f.size==x['bytes'];data=t.extractfile(f).read();assert hashlib.sha256(data).hexdigest()==x['sha256'];out=z.output/x['name']
  if out.exists():assert out.read_bytes()==data
  else:out.write_bytes(data)
print('EXACT_DVC_ASSET_PASS')
