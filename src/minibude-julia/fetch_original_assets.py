#!/usr/bin/env python3
"""Retrieve exactly the source DVC object into a new output directory."""
import argparse,hashlib,io,json,tarfile,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--archive',type=Path);a=p.parse_args();dest=a.output_dir.resolve();assert not dest.exists(),'Use a fresh directory; never overwrite measured assets'
m=json.loads(Path(__file__).with_name('original_asset_manifest.json').read_text());data=a.archive.read_bytes() if a.archive else urllib.request.urlopen(m['url'],timeout=120).read()
assert len(data)==m['archive_bytes'] and hashlib.md5(data).hexdigest()==m['DVC_md5'] and hashlib.sha256(data).hexdigest()==m['archive_sha256']
expected={f['archive_path']:f for f in m['files']};payload={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:*') as t:
 for name,record in expected.items():
  member=t.getmember(name);assert member.isfile();out=(dest/name).resolve();assert out.is_relative_to(dest);b=t.extractfile(member).read();assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256'];payload[out]=b
for out,b in payload.items():out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
print('Exact original archive and',len(payload),'files verified')
