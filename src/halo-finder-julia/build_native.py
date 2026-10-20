"""Build original full RCB tree/controller and owned observer, no OS changes."""
from pathlib import Path
import argparse,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--arch',default='sm_70');p.add_argument('--library',action='store_true');p.add_argument('--nvcc',default='nvcc');p.add_argument('--cc',default='gcc');a=p.parse_args();B=Path(__file__).resolve().parent;C=B/'configured_native';H=B/'reference_history';out=a.output.resolve()
def call(args):subprocess.run([str(x)for x in args],check=True)
flags=['-O3','-std=c++17','-arch='+a.arch,'-Xcompiler=-fPIC','-DRCB_UNTHREADED_BUILD','-DUSE_SERIAL_COSMO','-DID_64','-DPOSVEL_32','-DGRID_32','-DLONG_INTEGER','-I'+str(C),'-I'+str(H/'dfft'),'-I/usr/lib/x86_64-linux-gnu/openmpi/include']
with tempfile.TemporaryDirectory(prefix='halo_build_')as root:
 root=Path(root);objs=[]
 for name in ['RCOForceTree','ForceLaw','Partition','Timings','Timer']:
  obj=root/(name+'.o');call([a.nvcc,*flags,'-x','cu','-c',C/(name+'.cxx'),'-o',obj]);objs.append(obj)
 for src in [C/'bigchunk.c',H/'dfft/dims.c']:
  obj=root/(src.stem+'.o');call([a.cc,'-O3','-fPIC','-DUSE_SERIAL_COSMO','-I'+str(C),'-I'+str(H/'dfft'),'-I/usr/lib/x86_64-linux-gnu/openmpi/include','-c',src,'-o',obj]);objs.append(obj)
 obj=root/'host_driver.o';call([a.nvcc,*flags,*(['-DHALO_LIBRARY']if a.library else[]),'-c',B/'combined_cuda.cu','-o',obj]);objs.append(obj)
 call([a.nvcc,'-arch='+a.arch,*(['-shared','-Xcompiler=-fPIC']if a.library else[]),*objs,'-lm','-lrt','-o',out])
print(out)
