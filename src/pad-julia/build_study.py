from pathlib import Path
import subprocess,hashlib,json,sys,os
p=Path(__file__).resolve().parent;r=p.parents[1]
files=[p/'cpu_bridge.cpp',p/'cpu_bridge.h',r/'src/pad-cuda/host_task.cpp',r/'src/pad-cuda/kernel.h',r/'src/pad-cuda/support/common.h',r/'src/pad-cuda/support/partitioner.h']
compiler=subprocess.check_output(['g++','--version']).decode();flags=['-O3','-std=c++17','-fPIC','-shared','-pthread']
key=hashlib.sha256((compiler+str(flags)).encode()+b''.join(x.read_bytes() for x in files)).hexdigest();cache=r/'.porting-state/portable_julia/cpu_components/pad'/key;cache.mkdir(parents=True,exist_ok=True);lib=cache/'libpadcpu.so'
if not lib.exists():
 tmp=cache/('tmp'+str(os.getpid())+'.so');subprocess.run(['g++',*flags,str(p/'cpu_bridge.cpp'),str(r/'src/pad-cuda/host_task.cpp'),'-o',str(tmp)],check=True);tmp.replace(lib)
meta={'path':str(lib),'sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'compiler':compiler,'flags':flags,'source_key':key};(cache/'metadata.json').write_text(json.dumps(meta,indent=2));Path('cpu_component_path.txt').write_text(str(lib));print('STUDY_HOST_COMPONENT '+json.dumps(meta))
if sys.argv[1]=='native':subprocess.run(['nvcc','-O3','-std=c++17','-arch=sm_70',str(p/'study_cuda.cu'),str(r/'src/pad-cuda/kernel.cu'),'-L'+str(cache),'-lpadcpu','-Xlinker','-rpath','-Xlinker',str(cache),'-o','study_native'],check=True)
