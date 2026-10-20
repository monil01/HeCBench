#!/usr/bin/env python3
"""Recover only the two exact original DVC parameter files; CPU-only/no network."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();archive=Path(a.archive);raw=archive.read_bytes();assert len(raw)==79247 and hashlib.md5(raw).hexdigest()=='2d20b003f0064b7fd8fed7cb394a955e';out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
with tarfile.open(archive)as t:
 for name,n in [('MersenneTwister.dat',65536),('MersenneTwister.raw',262144)]:
  m=[m for m in t.getmembers()if m.isfile()and m.name.endswith('/'+name)];assert len(m)==1;data=t.extractfile(m[0]).read();assert len(data)==n;(out/name).write_bytes(data);rows.append({'file':name,'bytes':n,'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'archive_md5':hashlib.md5(raw).hexdigest(),'files':rows},indent=2))
