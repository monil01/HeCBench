#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,json
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--shared',action='store_true');a.add_argument('--gsl-prefix',type=Path,default=Path('/tmp/hecbench-stencil-ports-xlqc-deps/install'));a.add_argument('--arch',default='sm_70');a.add_argument('--nvcc',default='/usr/bin/nvcc');z=a.parse_args();O=P/'reference_history';I=P/'independent_oracle';sources=[P/'full_study.cu']+[O/x for x in ['cuda_rys_sp.cu','cuda_rys_dp.cu','basis.cc','scf.cc','int_lib/crys.cc','int_lib/cints.cc']]+[I/'prefixed_cints.cc',I/'eri_bridge.cc'];cmd=[z.nvcc,'-std=c++17','-O3','-rdc=true','-arch='+z.arch,'-I'+str(O),'-I'+str(O/'int_lib'),'-I'+str(z.gsl_prefix/'include'),*map(str,sources),'-L'+str(z.gsl_prefix/'lib'),'-lgsl','-lgslcblas','-lm','-lcrypto','-o',z.output]
if z.shared:cmd+=['-DXL_SHARED','--shared','-Xcompiler=-fPIC']
print('STUDY_BUILD_COMMAND',json.dumps(cmd),flush=True);subprocess.run(cmd,check=True)
