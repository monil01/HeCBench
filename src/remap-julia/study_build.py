from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
args=['nvcc','-O3','-std=c++17','-arch='+sys.argv[2],'--cudart','shared']
if sys.argv[1]=='shared':args+=['--shared','-Xcompiler=-fPIC',str(p/'study_vendor.cu'),'-o','libremap_vendor.so']
else:args += ['-DREMAP_ROOT="'+str(p)+'"',str(p/'study_native.cu'),str(p/'study_vendor.cu'),'-o','study_native']
print('BUILD',args,flush=True)
subprocess.run(args,check=True)
