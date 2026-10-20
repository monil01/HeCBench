#!/usr/bin/env python3
"""Recover only the original bundled NICSLU tree, rejecting external archive links."""
from pathlib import Path
import argparse,tarfile,hashlib,json,os
P=Path(__file__).resolve().parent;R=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path('/tmp/hecbench-stencil-ports-slu-original/src'));z=a.parse_args();out=z.output.resolve();archive=R/'src/slu-cuda/src/nicslu.tar.bz2';expected=json.load(open(P/'source_archive.json'))['original_vendor_archive']['sha256'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
out.mkdir(parents=True,exist_ok=True);records=[]
with tarfile.open(archive)as t:
 for m in t:
  p=Path(m.name);assert not p.is_absolute()and'..'not in p.parts
  if m.issym()or m.islnk():
   assert m.name=='nicslu/demo/matrix'and m.linkname=='/data/glu/matrix/','Unreviewedarchivelink';continue
  dest=out/p
  if m.isdir():dest.mkdir(parents=True,exist_ok=True);continue
  assert m.isfile(),'Unsupportedarchivemember';data=t.extractfile(m).read();dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.is_file()and dest.read_bytes()==data,'Refusemodifiedexistingoriginalfile'
  else:dest.write_bytes(data);dest.chmod(m.mode&0o777)
  records.append(dict(member=m.name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
print(json.dumps(dict(archive_sha256=expected,output=str(out),original_files=len(records),skipped_unused_external_alias='nicslu/demo/matrix')))
