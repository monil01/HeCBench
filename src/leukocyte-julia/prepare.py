#!/usr/bin/env python3
"""Unprivileged reproducible preparation; original dataset is a separate download."""
import argparse,hashlib,json,shutil,subprocess,tarfile,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path);parser.add_argument('--download',action='store_true');parser.add_argument('--cc',default='gcc');parser.add_argument('--cxx',default='g++');o=parser.parse_args()
video=p/'dataset_readiness/testfile.avi';video.parent.mkdir(exist_ok=True)
def digest(path,kind):
 h=hashlib.new(kind)
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
if not video.exists():
 archive=o.archive
 if archive is None and o.download:
  archive=video.parent/'leukocyte.tar.bz';url='https://hecbench-geekdude.s3.amazonaws.com/files/md5/79/470dcf4adeda242fec899bf0af6553'
  with urllib.request.urlopen(url,timeout=60) as source,archive.open('wb') as target:shutil.copyfileobj(source,target)
 if archive is None:parser.error('Supply --archive exact original DVC archive or --download; no synthetic replacement.')
 assert archive.stat().st_size==113303206 and digest(archive,'md5')=='79470dcf4adeda242fec899bf0af6553'
 with tarfile.open(archive,'r:*') as t:
  members=t.getmembers();assert len(members)==1 and members[0].name=='testfile.avi' and members[0].isfile()
  with t.extractfile(members[0]) as source,video.open('wb') as target:shutil.copyfileobj(source,target)
assert video.stat().st_size==184338944 and digest(video,'sha256')=='e3123dd9354293ab41703fc4f9e7c7cf8f7061e09b05d3ad9925941aa19020cc'
commands=[ [o.cc,'-O3','-shared','-fPIC',str(p/'study_video.c'),str(p/'../leukocyte-cuda/avilib.c'),'-o',str(p/'study_video.so')], ['make','-C',str(p/'corrected_native/meschach_lib'),'all','CC='+o.cxx,'CFLAGS=-O3 -std=c++17 -fPIC -x c++'] ]
for command in commands:print(json.dumps(command),flush=True);subprocess.run(command,check=True)
print('Original video and user-space CPU support ready. No GPU context or system package changes.')
