#!/usr/bin/env python3
"""Build CPU source-component bridge and optional unchanged native GPU entry."""
from pathlib import Path
import subprocess,sys,json
P=Path(__file__).resolve().parent;R=P.parent.parent;S=R/'src/logic-resim-cuda/Simulation/src';arch=sys.argv[1]if len(sys.argv)>1 else''
incs=['-I'+str(S/d)for d in ['gate','parser','sim','util','wave']]+['-I'+str(P)]
cpp=[str(P/'study_state.cpp'),str(P/'source_cpu_fallback.cpp')]+[str(S/f)for f in ['wave/Signal.cpp','gate/Gate.cpp','gate/DelayMgr.cpp','gate/TruthTable.cpp','util/util.cpp']]
cmd=['g++','-O3','-std=c++17','-shared','-fPIC','-pthread']+incs+cpp+['-o','study_host.so'];subprocess.run(cmd,check=True)
commands=[cmd]
if arch:
 cmd=['nvcc','-O3','-std=c++17','-arch='+arch,'-Xptxas=-v']+incs+[str(P/'study_native.cu'),'./study_host.so','-Xlinker','-rpath','-Xlinker',str(Path.cwd()),'-o','study_native'];commands.append(cmd);subprocess.run(cmd,check=True)
Path('build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
