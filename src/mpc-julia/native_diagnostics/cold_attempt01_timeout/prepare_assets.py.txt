#!/usr/bin/env python3
"""Cold CPU-only MPC bitcodec fixture generation, original exact DVC archive."""
from pathlib import Path
import hashlib,tarfile,tempfile,urllib.request,subprocess,json,shutil
P=Path(__file__).resolve().parent
URL='https://hecbench-geekdude.s3.amazonaws.com/files/md5/bb/136e68786cde4485a255d6b9dd6a3c'
def run():
 with tempfile.TemporaryDirectory(prefix='mpc-cold-') as td:
  td=Path(td);archive=td/'trace.tar.bz';h=hashlib.md5();n=0
  with urllib.request.urlopen(URL,timeout=90) as r,archive.open('wb') as f:
   while b:=r.read(1048576):
    n+=len(b)
    if n>275049163:raise ValueError('archive_size_guard')
    f.write(b);h.update(b)
  assert n==275049163 and h.hexdigest()=='bb136e68786cde4485a255d6b9dd6a3c'
  with tarfile.open(archive) as tf:
   m=tf.getmember('msg_sp.trace.out');assert m.isfile() and m.size==290105856
   with tf.extractfile(m) as f,(td/'real_prefix.bin').open('wb') as out:
    # Only first1M UInt64words needed by declared primary. Hash of fullarchive above binds complete sourceasset.
    data=f.read(1048576*8);assert len(data)==1048576*8;out.write(data)
  cmd=['g++','-O3','-std=c++17',str(P/'prepare_assets.cpp'),'-o',str(td/'prepare')];subprocess.run(cmd,check=True);subprocess.run([str(td/'prepare'),str(td/'real_prefix.bin')],cwd=P,check=True)
  paths=sorted((P/'fixtures').rglob('*.bin'))+[P/'cases_primary.txt',P/'cases_partial_corrected.txt'];print(json.dumps({str(q.relative_to(P)):{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in paths},indent=2))
if __name__=='__main__':run()
