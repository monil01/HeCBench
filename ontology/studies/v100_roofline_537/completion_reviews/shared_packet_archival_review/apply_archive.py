#!/usr/bin/env python3
"""PRIVATE root-review application script; no repository writes without --apply.
Archives approved static payloads and atomically appends four existing-schema
receipt entries. Never executes candidate code, mutates source, or sums tokens.
"""
from pathlib import Path
import argparse,fcntl,gzip,hashlib,json,os

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--proposal-root',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--apply',action='store_true');a=ap.parse_args()
 root=a.source_root.resolve();proposal=a.proposal_root.resolve();mp=proposal/'archive_manifest.json';raw=mp.read_bytes()
 if sha(raw)!=a.manifest_sha256:raise RuntimeError('Exact reviewed manifest SHA required')
 manifest=json.loads(raw);allowed=root/'ontology/studies/v100_roofline_537/completion_reviews';index=root/manifest['base_index']['path'];idxbytes=(proposal/manifest['proposed_index']['payload_relative']).read_bytes()
 if sha(idxbytes)!=manifest['proposed_index']['sha256']:raise RuntimeError('Proposed index changed')
 if index!=root/'ontology/studies/v100_roofline_537/SHARED_EFFORT_INDEX.json':raise RuntimeError('Index path outside exact scope')
 payloads=[]
 for e in manifest['files']:
  source=proposal/e['payload_relative'];target=root/e['destination']
  if not source.resolve().is_relative_to(proposal)or not target.resolve().is_relative_to(allowed):raise RuntimeError('Archive path escape')
  for parent in [target,*target.parents]:
   if parent==root.parent:break
   if parent.is_symlink():raise RuntimeError('Symlink archive destination forbidden')
  b=source.read_bytes()
  if sha(b)!=e['archived_sha256']or len(b)!=e['archived_bytes']:raise RuntimeError('Archive payload drift')
  original=gzip.decompress(b)if e['compression']=='gzip_mtime0'else b
  if sha(original)!=e['source_sha256']or len(original)!=e['source_bytes']:raise RuntimeError('Lossless source binding mismatch')
  if target.exists()and target.read_bytes()!=b:raise RuntimeError('Existing different archive preserved: '+str(target))
  payloads.append((target,b))
 if not a.apply:
  print(json.dumps({'not_applied':True,'archive_files':len(payloads),'new_receipts':len(manifest['new_index_receipts']),'manifest_sha256':a.manifest_sha256}));return
 # Root coordinator only: one locked base check before all shared archive/index writes.
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if sha(index.read_bytes())!=manifest['base_index']['sha256']:raise RuntimeError('Index changed: rebase/review proposal, never overwrite new receipts')
  for p,h in manifest['source_bindings'].items():
   if sha(Path(p).read_bytes())!=h:raise RuntimeError('Static source changed: '+p)
  old=json.loads(index.read_bytes());new=json.loads(idxbytes)
  if new['root_window']!=old['root_window']or new['sum_prohibited_reason']!=old['sum_prohibited_reason']:raise RuntimeError('Open-root/no-sum policy changed')
  if new['receipts']!=sorted(old['receipts']+manifest['new_index_receipts'],key=lambda r:r['path']):raise RuntimeError('Unexpected index mutation')
  for target,b in payloads:
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():
    with target.open('xb')as out:out.write(b)
   if target.read_bytes()!=b:raise RuntimeError('Archive write verification mismatch')
  temp=index.with_name(index.name+'.'+a.manifest_sha256[:16]+'.tmp')
  with temp.open('xb')as out:out.write(idxbytes);out.flush();os.fsync(out.fileno())
  temp.replace(index)
 print(json.dumps({'applied':True,'archive_files':len(payloads),'new_receipts':len(manifest['new_index_receipts']),'index_sha256':sha(index.read_bytes()),'token_sum':None}))
if __name__=='__main__':main()
