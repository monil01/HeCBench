from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-pthread',str(p/'study_cpu.cpp'),'-o','libsc_cpu.so'],check=True)
if sys.argv[1]=='native':subprocess.run(['nvcc','-O3','-std=c++17','-arch='+sys.argv[2],'-DSC_ROOT="'+str(p)+'"',str(p/'study_native.cu'),'-L.','-lsc_cpu','-Xlinker=-rpath=.', '-Xcompiler=-pthread','-o','study_native'],check=True)
