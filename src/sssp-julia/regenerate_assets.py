#!/usr/bin/env python3
"""Reproduce CPU graph/oracle assets and common CPU-only worker library, no GPU."""
import argparse,pathlib,hashlib,json,urllib.request,shutil,tarfile,subprocess,tempfile
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--archive',type=pathlib.Path);a.add_argument('--proof',type=pathlib.Path);z=a.parse_args();o=z.output.resolve();o.mkdir(parents=True,exist_ok=True);pro=json.load(open(P/'native_diagnostics/dataset_provenance.json'));archive=z.archive or(o/'data.tar.gz')
 if not archive.exists():
  with urllib.request.urlopen(pro['url'])as q,open(archive,'wb')as w:shutil.copyfileobj(q,w)
 assert archive.stat().st_size==pro['bytes']and hashlib.md5(archive.read_bytes()).hexdigest()==pro['md5']and sha(archive)==pro['sha256']
 with tempfile.TemporaryDirectory(prefix='sssp-cpu-regenerate-')as t:
  t=pathlib.Path(t)
  with tarfile.open(archive)as tf:
   assert all(not pathlib.PurePosixPath(x).is_absolute()and'..'not in pathlib.PurePosixPath(x).parts for x in tf.getnames());tf.extractall(t,filter='data')
  subprocess.run(['python3',str(P/'generate_assets.py'),'--dataset-root',str(t),'--output',str(o)],check=True)
 cmd=['g++','-O3','-std=c++17','-shared','-fPIC','-pthread','-x','c++',str(P/'host_bridge.cpp'),str(R/'src/sssp-cuda/kernel.cu'),'-o',str(o/'libsssp_host.so')];subprocess.run(cmd,check=True)
 expected=json.load(open(P/'fixture_manifest.json'))['files'];rows=[]
 for name,m in expected.items():
  f=o/name;assert f.stat().st_size==m['bytes']and sha(f)==m['sha256'];rows.append(dict(name=name,bytes=f.stat().st_size,sha256=sha(f)))
 expected_library=next(x for x in json.load(open(P/'cold_regeneration_proof.json'))['files']if x['name']=='libsssp_host.so');assert sha(o/'libsssp_host.so')==expected_library['sha256'],'CPUsharedlibdifferentbytes; inspectcompiler/platform,donotsilentlyrebindmeasuredlibrary'
 rows.append(dict(name='libsssp_host.so',bytes=(o/'libsssp_host.so').stat().st_size,sha256=sha(o/'libsssp_host.so')));v=dict(status='byte_exact',CPU_only=True,GPU_execution=False,files=rows,bytes=sum(x['bytes']for x in rows),original_DVC_archive=dict(bytes=archive.stat().st_size,sha256=sha(archive),md5=pro['md5'],url=pro['url']),host_library_build=cmd,compiler_version=subprocess.check_output(['g++','--version'],text=True).splitlines(),irreplaceable_binary_assets=[])
 if z.proof:z.proof.write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(dict(status='byte_exact',files=len(rows),bytes=v['bytes'])))
if __name__=='__main__':main()
