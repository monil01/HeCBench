#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,urllib.request,json
URL='https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2';SIZE=21902461;SHA='26d95cd35c9c51d1b25f5741dc95c0482278aab85a90f1920aa6152616b23d41'
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
if not args.output.exists():
 with urllib.request.urlopen(URL,timeout=120)as src,args.output.open('wb')as dst:
  n=0
  for b in iter(lambda:src.read(1<<20),b''):
   n+=len(b);assert n<=SIZE;dst.write(b)
raw=args.output.read_bytes();assert len(raw)==SIZE and hashlib.sha256(raw).hexdigest()==SHA;print(json.dumps({'url':URL,'bytes':SIZE,'sha256':SHA}))
