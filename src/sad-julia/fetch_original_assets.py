#!/usr/bin/env python3
"""Retrieve only the exact declared original DVC archive, preserve SHA provenance."""
from pathlib import Path
import hashlib,urllib.request,tarfile,json
P=Path(__file__).resolve().parent;manifest=json.loads((P/'original_asset_manifest.json').read_text());archive=P/'data.tar.gz'
if not archive.exists():
 with urllib.request.urlopen('https://hecbench-geekdude.s3.amazonaws.com/files/md5/8e/70751dd405b713f2b7dad2fec7a27b',timeout=60) as f:b=f.read(318793)
 assert len(b)==318792 and hashlib.md5(b).hexdigest()==manifest['DVC_md5'] and hashlib.sha256(b).hexdigest()==manifest['archive_sha256'];archive.write_bytes(b)
b=archive.read_bytes();assert len(b)==318792 and hashlib.md5(b).hexdigest()==manifest['DVC_md5'] and hashlib.sha256(b).hexdigest()==manifest['archive_sha256']
assets=P/'original_assets';assets.mkdir(exist_ok=True);expected={Path(e['archive_path']).name:e for e in manifest['files']}
with tarfile.open(archive) as t:
 for member in t.getmembers():
  if not member.isfile():continue
  name=Path(member.name).name;assert name in expected;data=t.extractfile(member).read();e=expected[name];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'];(assets/name).write_bytes(data)
for name,e in expected.items():assert hashlib.sha256((assets/name).read_bytes()).hexdigest()==e['sha256']
print('Exact original DVC archive and three assets verified.')
