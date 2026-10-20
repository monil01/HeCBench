#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,json
P=Path(__file__).resolve().parent;R=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--gsl-prefix',type=Path,default=Path('/tmp/hecbench-stencil-ports-xlqc-deps/install'));a.add_argument('--nvcc',default='/usr/bin/nvcc');a.add_argument('--arch',default='sm_70');z=a.parse_args();O=R/'src/xlqc-cuda';sources=[O/x for x in ['main.cu','cuda_rys_sp.cu','cuda_rys_dp.cu','basis.cc','scf.cc','int_lib/crys.cc','int_lib/cints.cc']];cmd=[z.nvcc,'-std=c++17','-O3','-rdc=true','-arch='+z.arch,'-I'+str(z.gsl_prefix/'include'),*map(str,sources),'-L'+str(z.gsl_prefix/'lib'),'-lgsl','-lgslcblas','-lm','-o',z.output];print('ORIGINAL_BUILD_COMMAND',json.dumps(cmd),flush=True);subprocess.run(cmd,check=True)
