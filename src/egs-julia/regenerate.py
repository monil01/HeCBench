"""Regenerate exact original HeCBench DVC scientific assets; execute no downloaded code."""
from pathlib import Path
import argparse,urllib.request,hashlib,tarfile,json,io
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();proof=json.loads((Path(__file__).resolve().parent/'original_dvc_asset_provenance.json').read_text());raw=urllib.request.urlopen(proof['url'],timeout=90).read(proof['bytes']+1);assert len(raw)==proof['bytes'] and hashlib.md5(raw).hexdigest()==proof['expected_md5'] and hashlib.sha256(raw).hexdigest()==proof['sha256'];a.data_root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
 for row in proof['entries']:
  m=t.getmember(row['archive_entry']);assert m.isfile() and m.size==row['bytes'];b=t.extractfile(m).read();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];(a.data_root/Path(row['archive_entry']).name).write_bytes(b)
print(json.dumps({'original_DVC_md5':proof['expected_md5'],'archive_sha256':proof['sha256'],'files':len(proof['entries']),'data_root':str(a.data_root.resolve())}))
