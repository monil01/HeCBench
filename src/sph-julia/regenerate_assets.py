#!/usr/bin/env python3
"""CPU-only fixture/oracle reproduction; compiles CUDA source but launches no GPU."""
import argparse,hashlib,json,pathlib,subprocess,tempfile
P=pathlib.Path(__file__).resolve().parent;S=P.parent/'sph-cuda'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--nvcc',default='/usr/bin/nvcc');a.add_argument('--cxx',default='g++');a.add_argument('--proof',type=pathlib.Path);z=a.parse_args();out=z.output.resolve();out.mkdir(parents=True,exist_ok=True);commands=[]
 with tempfile.TemporaryDirectory(prefix='sph-cpu-regenerate-') as tmp:
  tmp=pathlib.Path(tmp);cmd=[z.nvcc,'-O3','-std=c++17','-arch=sm_70',str(P/'generate_inputs.cu'),str(S/'geometry.cu'),str(S/'fileio.cu'),'-o',str(tmp/'generate_inputs')];commands.append(cmd);subprocess.run(cmd,check=True)
  cmd=[str(tmp/'generate_inputs'),str(out)];commands.append(cmd);subprocess.run(cmd,check=True)
  cmd=[z.cxx,'-O3','-std=c++17','-fopenmp','-ffp-contract=off',str(P/'reference_cpu.cpp'),'-o',str(tmp/'reference_cpu')];commands.append(cmd);subprocess.run(cmd,check=True)
  for case in ['default','ballistic']:
   cmd=[str(tmp/'reference_cpu'),str(out),case];commands.append(cmd);subprocess.run(cmd,check=True)
 files=sorted(out.glob('*.bin'));manifest=json.load(open(P/'asset_hashes.json'));rows=[]
 for f in files:
  h=sha(f);assert h==manifest[f.name]['sha256'] and f.stat().st_size==manifest[f.name]['bytes'],f.name
  rows.append(dict(name=f.name,sha256=h,bytes=f.stat().st_size))
 assert set(x['name']for x in rows)==set(manifest)
 proof=dict(status='byte_exact',cpu_only_execution=True,device_compilation_but_no_gpu_launch=True,commands=commands,files=rows,bytes=sum(x['bytes']for x in rows),irreplaceable_assets=[],source_hashes={str(f):sha(f)for f in [P/'generate_inputs.cu',P/'reference_cpu.cpp',S/'fluid.cu',S/'geometry.cu',S/'fileio.cu',S/'common.h']},compiler_versions={x:subprocess.check_output([x,'--version'],text=True).splitlines()for x in [z.nvcc,z.cxx]})
 if z.proof:z.proof.write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps({'status':'byte_exact','files':len(rows),'bytes':proof['bytes']}))
if __name__=='__main__':main()
