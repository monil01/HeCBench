#!/usr/bin/env python3
"""Build the pinned GSL dependency in a fresh unprivileged private prefix."""
from pathlib import Path
import argparse,json,urllib.request,hashlib,tarfile,subprocess,os
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--jobs',type=int,default=4);z=a.parse_args();root=z.root.resolve();root.mkdir(parents=True,exist_ok=True);meta=json.load(open(P/'dependency_source.json'));tar=root/'gsl-2.8.tar.gz'
assert not(root/'gsl-2.8').exists()and not(root/'install').exists(),'Use a fresh dependency root'
if tar.exists():b=tar.read_bytes()
else:
 with urllib.request.urlopen(meta['url'],timeout=90)as r:assert r.status==200;b=r.read(meta['bytes']+1)
 assert len(b)==meta['bytes'];tar.write_bytes(b)
assert len(b)==meta['bytes']and hashlib.sha256(b).hexdigest()==meta['sha256']
with tarfile.open(tar)as t:t.extractall(root,filter='data')
source=root/'gsl-2.8';env=os.environ.copy();env['CFLAGS']='-O2 -fPIC';commands=[['./configure','--prefix='+str(root/'install'),'--disable-shared'],['make','-j'+str(z.jobs)],['make','install']]
for i,cmd in enumerate(commands):
 with(root/f'build_step{i}.log').open('w')as log:subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
receipt={'source':meta,'commands':commands,'prefix':str(root/'install'),'scope':'All output under the caller-selected unprivileged root; no system package/environment changes.','static_libraries':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in(root/'install/lib').glob('*.a')}};(root/'dependency_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(root/'install')
