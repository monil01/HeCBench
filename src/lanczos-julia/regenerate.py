#!/usr/bin/env python3
"""Optional exact original asset retrieval; named GPU fixture requires no download."""
import argparse,pathlib,urllib.request,hashlib,tarfile,json
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=pathlib.Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);dest=a.output_dir/'data.tar.bz';size=23191260;url='https://hecbench-geekdude.s3.amazonaws.com/files/md5/2b/9896a988b45223138a88317fa4a7e3'
with urllib.request.urlopen(url,timeout=60) as src,dest.with_suffix('.download').open('wb') as out:
 total=0
 while True:
  chunk=src.read(1<<20)
  if not chunk:break
  total+=len(chunk)
  if total>size:raise ValueError('DVCsizeguard')
  out.write(chunk)
f=dest.with_suffix('.download');assert f.stat().st_size==size and hashlib.md5(f.read_bytes()).hexdigest()=='2b9896a988b45223138a88317fa4a7e3';f.replace(dest)
with tarfile.open(dest) as t:
 for m in t.getmembers():
  if not m.isfile():continue
  target=(a.output_dir/m.name).resolve();assert target.is_relative_to(a.output_dir.resolve()) and m.size<250000000;target.parent.mkdir(parents=True,exist_ok=True)
  with t.extractfile(m) as s,target.open('wb') as o:o.write(s.read())
print(json.dumps({'archive':str(dest),'bytes':size,'MD5':'2b9896a988b45223138a88317fa4a7e3','sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'role':'Original800kCLI diagnostic asset, notessentialtonamed257nodefourwaystudy'}))
