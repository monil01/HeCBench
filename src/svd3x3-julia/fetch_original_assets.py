#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,urllib.request,tarfile
P=Path(__file__).resolve().parent;m=json.loads((P/'original_asset_manifest.json').read_text());a=P/'dataset.tar.bz'
if not a.exists():
 with urllib.request.urlopen(m['url'],timeout=60)as f:b=f.read(m['archive_bytes']+1)
 assert len(b)==m['archive_bytes']and hashlib.md5(b).hexdigest()==m['DVC_md5']and hashlib.sha256(b).hexdigest()==m['archive_sha256'];a.write_bytes(b)
b=a.read_bytes();assert len(b)==m['archive_bytes']and hashlib.md5(b).hexdigest()==m['DVC_md5']and hashlib.sha256(b).hexdigest()==m['archive_sha256'];out=P/'original_assets';out.mkdir(exist_ok=True)
with tarfile.open(a)as t:
 for e in m['files']:
  q=t.getmember(e['archive_path']);assert q.isfile();b=t.extractfile(q).read();assert len(b)==e['bytes']and hashlib.sha256(b).hexdigest()==e['sha256'];(out/e['name']).write_bytes(b)
print('Exact original DVC archive and dataset hashes verified')
