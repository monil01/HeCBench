#!/usr/bin/env python3
"""CPU-only regeneration of the preselected, balanced source-DVC MNIST subset.
No network, package install, CUDA, privileged operations or model prediction.
Native initial parameters are a separately committed actual-readback fixture.
"""
import argparse,hashlib,json,struct,zipfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();archive=Path(a.archive);raw=archive.read_bytes();assert len(raw)==11595448 and hashlib.md5(raw).hexdigest()=='53c5aacb1b0e259154ef170e15bedab8';out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 with zipfile.ZipFile(archive) as z:
  for prefix,perclass in [('train',10),('t10k',2)]:
   names=[prefix+'-images.idx3-ubyte',prefix+'-labels.idx1-ubyte'];members=[]
   for name in names:
    matches=[x for x in z.namelist() if x.endswith('/'+name) or x==name];assert len(matches)==1;members.append(z.read(matches[0]))
   im,labels=members;magic,n,h,w=struct.unpack_from('>4I',im);magic2,n2=struct.unpack_from('>2I',labels);assert(magic,magic2,h,w)==(2051,2049,28,28) and n==n2 and len(im)==16+784*n and len(labels)==8+n
   counts=[0]*10;selected=[]
   for i,label in enumerate(labels[8:]):
    assert 0<=label<=9
    if counts[label]<perclass:selected.append(i);counts[label]+=1
   assert counts==[perclass]*10;newimages=struct.pack('>4I',2051,len(selected),28,28)+b''.join(im[16+i*784:16+(i+1)*784] for i in selected);newlabels=struct.pack('>2I',2049,len(selected))+bytes(labels[8+i] for i in selected)
   for name,data in zip(names,(newimages,newlabels)):(out/name).write_bytes(data);rows.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 print(json.dumps({'archive_md5':hashlib.md5(raw).hexdigest(),'generated':rows},indent=2))
if __name__=='__main__':main()
