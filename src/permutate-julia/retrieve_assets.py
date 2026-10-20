#!/usr/bin/env python3
"""Exact public DVC asset retrieval; bounded bytes, no package/system changes."""
import argparse,pathlib,urllib.request,tarfile,hashlib,json,datetime,io
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=pathlib.Path);a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parents[1];out=a.output_dir or R/'.porting-state/porting_logs/permutate-julia-shared/assets';out.mkdir(parents=True,exist_ok=True);e=json.loads((B/'asset_provenance.json').read_text());archive=out/'dataset.tar.bz';status=None
if archive.exists():raw=archive.read_bytes()
else:
 with urllib.request.urlopen(e['url'],timeout=30)as f:
  status=f.status;raw=f.read(e['actual_archive_bytes']+1)
assert len(raw)==e['actual_archive_bytes'] and hashlib.md5(raw).hexdigest()==e['declared_md5'] and hashlib.sha256(raw).hexdigest()==e['actual_archive_sha256']
if not archive.exists():q=archive.with_suffix('.tmp');q.write_bytes(raw);q.replace(archive)
with tarfile.open(fileobj=io.BytesIO(raw))as t:
 m=t.getmember(e['member']['name']);assert m.size==e['member']['bytes'];data=t.extractfile(m).read()
assert len(data)==1000000 and hashlib.sha256(data).hexdigest()==e['member']['sha256']
p=out/'truerand_1bit.bin'
if p.exists():assert p.read_bytes()==data
else:q=p.with_suffix('.tmp');q.write_bytes(data);q.replace(p)
record=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),url=e['url'],HTTP_status_this_invocation=status,used_existing_exact_archive=status is None,archive_sha256=hashlib.sha256(raw).hexdigest(),member_sha256=hashlib.sha256(data).hexdigest(),scope='New reproducibility invocation; original historical GET HTTP status was not retained and is not backfilled.')
(out/'retrieval_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json').write_text(json.dumps(record,indent=2)+'\n');print(p)
