#!/usr/bin/env python3
"""CPU-only cold regeneration; pinned upstream download is optional, size/hash guarded."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile,urllib.request
P=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--dataset',type=Path);a.add_argument('--report',type=Path);x=a.parse_args();out=x.output.resolve();out.mkdir(parents=True,exist_ok=True);prov=json.load(open(P/'dataset_provenance.json'));dataset=x.dataset
 if dataset is None:
  dataset=out/'QCLOUDf48.bin.f32.errctrl.1e-3'
  if not dataset.exists():
   n=0
   with urllib.request.urlopen(prov['url'],timeout=120)as r,open(dataset,'wb')as f:
    while b:=r.read(1<<20):
     n+=len(b)
     if n>prov['bytes']:raise RuntimeError('download size guard exceeded')
     f.write(b)
 assert dataset.stat().st_size==prov['bytes']and sha(dataset)==prov['sha256'],'dataset identity'
 recipe=P/'generate_oracle.cpp';versions=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
 with tempfile.TemporaryDirectory(prefix='lzss-cpu-')as t:
  binary=Path(t)/'generate_oracle';cmd=['g++','-O3','-std=c++17',str(recipe),'-o',str(binary)];subprocess.run(cmd,check=True);generation=subprocess.check_output([str(binary),str(dataset.resolve()),str(out)],text=True)
 expected=json.load(open(P/'fixture_provenance.json'))['files'];files={}
 for name,e in expected.items():
  p=out/name;files[name]={'bytes':p.stat().st_size,'sha256':sha(p)};assert files[name]==e,(name,files[name],e)
 report=dict(cpu_only=True,no_GPU_calls=True,dataset_sha256=prov['sha256'],generator_sha256=sha(recipe),compiler=versions,command=cmd,stdout=generation,files=files,all_exact=True)
 if x.report:x.report.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
