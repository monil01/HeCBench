"""Fetch only hash-pinned public scientific data; no downloaded code executes."""
from pathlib import Path
import argparse,urllib.request,hashlib,zipfile,json,io
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent;proof=json.loads((here/'asset_provenance.json').read_text());raw=urllib.request.urlopen(proof['download_url'],timeout=90).read();assert len(raw)==proof['archive_bytes'] and hashlib.sha256(raw).hexdigest()==proof['archive_sha256'];a.data_root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw))as z:
 for row in proof['scientific_data_files']:
  b=z.read(row['upstream_entry']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];original=Path(row['local_original']).name;(a.data_root/original).write_bytes(b);(a.data_root/row['source_requested_alias']).write_bytes(b)
print(json.dumps({'archive_sha256':proof['archive_sha256'],'scientific_data_files':len(proof['scientific_data_files']),'data_root':str(a.data_root.resolve()),'no_downloaded_code_executed':True}))
