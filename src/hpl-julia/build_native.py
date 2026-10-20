#!/usr/bin/env python3
import pathlib,subprocess,json,os,sys
p=pathlib.Path(__file__).resolve().parent
out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'study_native').resolve();obj=out.parent/'hpl_objects';obj.mkdir(parents=True,exist_ok=True)
inc=p/'reference_history/src/hpl-2.3/include'
flags=['-O3','-fPIC','-DSTDC_HEADERS','-DHPL_CALL_CBLAS','-DAdd_','-I'+str(inc),'-I'+str(p)]
files=[p/f for f in json.loads((p/'source_list.json').read_text())]+[p/'blas_integer_bridge.c']
objects=[]
for i,f in enumerate(files):
 o=obj/(str(i)+'_'+f.stem+'.o');objects.append(str(o))
 if not o.exists() or o.stat().st_mtime<f.stat().st_mtime:subprocess.run(['mpicc',*flags,'-c',str(f),'-o',str(o)],check=True)
lib=pathlib.Path('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/libopenblas64_.so')
subprocess.run(['ar','rcs',str(obj/'libhpl_original.a'),*objects],check=True)
print('HPL original CPU archive ready',len(objects),flush=True)

subprocess.run(['nvcc','-g','-O3','-std=c++17','-arch=sm_70','-DHPL_CALL_CBLAS','-DAdd_','-I'+str(inc),'-I'+str(p),'-I/usr/lib/x86_64-linux-gnu/openmpi/include','-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi',str(p/'host_driver.cu'),str(obj/'libhpl_original.a'),str(lib),'-Xlinker','-rpath','-Xlinker',str(lib.parent),'-lcublas','-lmpi','-o',str(out)],check=True)
