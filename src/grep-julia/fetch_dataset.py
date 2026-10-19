#!/usr/bin/env python3
"""Recover the exact declared DVC object; reject wrong sizes/hashes and unsafe members."""
from pathlib import Path
import urllib.request,hashlib,tarfile
P=Path(__file__).resolve().parent;R=P.parent.parent;D=R/'.porting-state/diagnostics/grep/external_dataset';D.mkdir(parents=True,exist_ok=True);f=D/'testcase.tar.gz'
if not f.exists():
 with urllib.request.urlopen('https://hecbench-geekdude.s3.amazonaws.com/files/md5/1b/e484e521e97149658b113eb2b3848d',timeout=30)as response:
  raw=response.read(630880)
 assert len(raw)==630879 and hashlib.md5(raw).hexdigest()=='1be484e521e97149658b113eb2b3848d';f.write_bytes(raw)
raw=f.read_bytes();assert len(raw)==630879 and hashlib.md5(raw).hexdigest()=='1be484e521e97149658b113eb2b3848d'
with tarfile.open(f)as t:
 for m in t.getmembers():assert(D/m.name).resolve().is_relative_to(D.resolve())and not m.issym()and not m.islnk()
 t.extractall(D,filter='data')
print(D/'testcases/romeojuliet.txt')
