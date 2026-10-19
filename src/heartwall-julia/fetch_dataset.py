#!/usr/bin/env python3
"""Retrieve exactly the source-declared DVC object; bound size/hash before safe extraction."""
from pathlib import Path
import argparse,hashlib,json,tarfile,urllib.request
MD5='f247f879ea9b4f47729e15a959ea117d';SIZE=13133765
URL=f'https://hecbench-geekdude.s3.amazonaws.com/files/md5/{MD5[:2]}/{MD5[2:]}'
a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,required=True);args=a.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);archive=args.output_dir/'heartwall.tar.bz'
if not archive.exists():
 with urllib.request.urlopen(URL,timeout=120)as response,archive.open('wb')as out:
  total=0
  for b in iter(lambda:response.read(1<<20),b''):
   total+=len(b)
   if total>SIZE:raise RuntimeError('Declared object size exceeded')
   out.write(b)
raw=archive.read_bytes();assert len(raw)==SIZE and hashlib.md5(raw).hexdigest()==MD5
with tarfile.open(archive)as t:
 assert set(t.getnames())=={'input.txt','test.avi'};t.extractall(args.output_dir,filter='data')
print(json.dumps({'url':URL,'bytes':SIZE,'md5':MD5,'sha256':hashlib.sha256(raw).hexdigest(),'files':['input.txt','test.avi']}))
