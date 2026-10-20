#!/usr/bin/env python3
"""CPU-only cold reproduction. Exact DVC object, source scheduler, independent DP."""
import argparse,hashlib,json,subprocess,tarfile,tempfile,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
URL='https://hecbench-geekdude.s3.amazonaws.com/files/md5/1a/9d6bde2cf5246bc7cc955d6c861169'
def run():
 with tempfile.TemporaryDirectory(prefix='minimap2-cold-') as td:
  td=Path(td); archive=td/'in.tar.gz'
  with urllib.request.urlopen(URL,timeout=60) as r,archive.open('wb') as f:
   n=0
   while True:
    b=r.read(1048576)
    if not b:break
    n+=len(b)
    if n>22646314:raise ValueError('size guard')
    f.write(b)
  data=archive.read_bytes();assert len(data)==22646314 and hashlib.md5(data).hexdigest()=='1a9d6bde2cf5246bc7cc955d6c861169'
  with tarfile.open(archive) as t: lines=t.extractfile('in-1k.txt').read().decode().splitlines()
  selected=[];i=0
  while i<len(lines):
   if not lines[i].strip():i+=1;continue
   begin=i;n=int(lines[i].split()[0]);i+=n+2
   if len(selected)<128 and 64<=n<=3072:selected.append(lines[begin:i])
  assert len(selected)==128
  (P/'study_reads.txt').write_text('\n'.join(line for chunk in selected for line in chunk)+'\n')
  root=P.parent.parent
  cmd=['g++','-O3','-std=c++17','-ffp-contract=off','-I',str(root/'src/minimap2-cuda/include'),str(P/'prepare_assets.cpp'),'-o',str(td/'prepare')]
  subprocess.run(cmd,check=True);subprocess.run([str(td/'prepare')],cwd=P,check=True)
  print(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [P/'study_reads.txt',P/'sizes.txt',*sorted(P.glob('*.bin'))]},indent=2))
if __name__=='__main__':run()
