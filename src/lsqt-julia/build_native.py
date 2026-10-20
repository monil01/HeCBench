#!/usr/bin/env python3
from pathlib import Path
import argparse,json,subprocess
p=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('--output',default='study_native');parser.add_argument('--nvcc',default='/usr/bin/nvcc');a=parser.parse_args();files=[p/'study_native_driver.cu']+[p/'instrumented_native'/n for n in ['lsqt.cu','sigma.cu','vector.cu','hamiltonian.cu','model.cu','model_general.cu','model_lattice.cu','anderson.cu','charge.cu']];objects=[];commands=[]
for i,source in enumerate(files):
 target=Path.cwd()/f'lsqt_native_{i}.o';command=[a.nvcc,'-O3','-std=c++17','-arch=sm_70','-use_fast_math','-DDEBUG','-x','cu','-c',str(source),'-o',str(target)];commands.append(command);print('NATIVE_BUILD_COMMAND '+json.dumps(command),flush=True);subprocess.run(command,check=True);objects.append(str(target))
command=[a.nvcc]+objects+['-o',a.output];commands.append(command);print('NATIVE_BUILD_COMMAND '+json.dumps(command),flush=True);subprocess.run(command,check=True);Path('native_build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
