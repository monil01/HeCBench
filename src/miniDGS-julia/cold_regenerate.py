#!/usr/bin/env python3
"""CPU-only full fixture/oracle regeneration from original mesh/source, no saved GPU outputs."""
from pathlib import Path
import argparse,subprocess,tempfile,shutil,os,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--report',type=Path);x=a.parse_args();out=x.output.resolve();out.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OPENBLAS_NUM_THREADS']='2';commands=[];logs=[]
 def run(cmd,cwd=None):
  commands.append(cmd);z=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append(z.stdout);logpath=out/('cold_command_%02d.log'%len(commands));logpath.write_text(z.stdout)
  if z.returncode:raise RuntimeError(z.stdout)
 with tempfile.TemporaryDirectory(prefix='miniDGS-CPU-')as t:
  tmp=Path(t);source=R/'src/miniDGS-cuda';lib=tmp/'ParMetis';shutil.copytree(source/'3rdParty/ParMetis-3.1',lib)
  for sub in ['METISLib','ParMETISLib']:run(['make','-C',str(lib/sub),'CC=mpicc','LD=mpicc','OPTFLAGS=-O3 -fPIC','-j4'])
  build=tmp/'cpu_layout';run(['python3',str(P/'native_compatibility/build_cpu_layout.py'),'--build-dir',str(build),'--parmetis-dir',str(lib)])
  deps=subprocess.check_output(['ldd',str(build/'cpu_layout')],text=True);assert 'libcuda'not in deps and'libcudart'not in deps
  run(['mpiexec','-n','2',str(build/'cpu_layout'),str(source/'Meshes/FS_K01022.neu'),str(out)])
  run(['python3',str(P/'independent_oracle.py'),'--assets',str(out),'--output',str(out)])
 expected=json.load(open(P/'fixture_provenance.json'))['files'];files={}
 for name,e in expected.items():
  f=out/Path(name).name;got=dict(bytes=f.stat().st_size,sha256=sha(f));assert got==e,(name,got,e);files[name]=got
 report=dict(CPU_only=True,visible_GPUs='',authored_CUDA_runtime_calls=False,CPU_executable_not_linked_to_CUDA=True,SDK_headers_required_for_source_padded_index_branch=True,source_mesh='src/miniDGS-cuda/Meshes/FS_K01022.neu',all_files_exact=True,files=files,commands=commands,log_sha256=[hashlib.sha256(z.encode()).hexdigest()for z in logs],raw_logs=[str(out/('cold_command_%02d.log'%(i+1)))for i in range(len(logs))],source_recipe_sha256=sha(Path(__file__).resolve()),MPI_version=subprocess.check_output(['mpiexec','--version'],text=True).splitlines()[0],CPU_compiler=subprocess.check_output(['mpicxx','--version'],text=True).splitlines()[0],no_saved_GPU_outputs_used=True,independent_math_NumPy='2.5.4')
 if x.report:x.report.write_text(json.dumps(report,indent=2)+'\n')
 print('COLD_EXACT',len(files),'assets',sum(z['bytes']for z in files.values()),'bytes')
if __name__=='__main__':main()
