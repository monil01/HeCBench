#!/usr/bin/env python3
"""CPU-only original mesh preprocessing and source-host packing, never a fake GPU check."""
from pathlib import Path
import re,argparse,subprocess,json,hashlib
P=Path(__file__).resolve().parents[1];R=P.parents[1];S=R/'src/miniDGS-cuda';a=argparse.ArgumentParser();a.add_argument('--build-dir',type=Path,required=True);a.add_argument('--parmetis-dir',type=Path,required=True);x=a.parse_args();B=x.build_dir.resolve();B.mkdir(parents=True,exist_ok=True)
s=(S/'src/MaxwellsKernel3d.cu').read_text();init=s[s.index('double InitGPU3d('):s.index('__global__ void MaxwellsGPU_VOL_Kernel3D')];setter=s[s.index('void gpu_set_data3d('):s.index('void gpu_get_data3d(')];body=init+setter
body=body.replace('InitGPU3d','study_init_host_layout').replace('gpu_set_data3d','study_set_host_fields')
body=re.sub(r'checkError\(cudaBindTexture\(0,\s*t_\w+,\s*\w+,\s*\w+\)\);','/* CPU-only: no texture binding exists. */',body)
body=re.sub(r'cudaMalloc\s*\(\(void\*\*\)\s*&\(?([\w>.-]+)\)?,\s*sz\);',r'\1=(decltype(\1))malloc(sz);',body)
body=re.sub(r'cudaMemcpy\(\s*(.*?),\s*(.*?),\s*(.*?),\s*cudaMemcpyHostToDevice\);',r'memcpy(\1,\2,\3);',body)
assert not re.search(r'\bcuda\w+\s*\(',body),body
preamble='''#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <cmath>\n#include <mpi.h>\nextern "C" {\n#include "fem.h"\n}\nstatic float *c_LIFT,*c_DrDsDt,*c_surfinfo,*c_vgeo,*c_Q,*c_partQ,*c_rhsQ,*c_resQ,*c_tmp;\nstruct float4{float x,y,z,w;};\n'''
export=r'''
static const char*OUTPUT;
static void dump(const char*name,int rank,const void*data,size_t bytes){char path[4096];snprintf(path,sizeof(path),"%s/rank%d_%s.bin",OUTPUT,rank,name);FILE*f=fopen(path,"wb");if(!f||fwrite(data,1,bytes,f)!=bytes)exit(9);fclose(f);}
int main(int argc,char**argv){MPI_Init(&argc,&argv);if(argc!=3)MPI_Abort(MPI_COMM_WORLD,3);OUTPUT=argv[2];int maxNv;Mesh*m=ReadMesh3d(argv[1]);LoadBalance3d(m);FacePair3d(m,&maxNv);StartUp3d(m);study_init_host_layout(m,6);size_t valid=m->K*p_Np;double*zero=(double*)calloc(valid,8),*Ez=(double*)malloc(valid*8);for(int k=0;k<m->K;k++)for(int n=0;n<p_Np;n++)Ez[k*p_Np+n]=cos(M_PI*m->x[k][n])*cos(M_PI*m->y[k][n])*cos(M_PI*m->z[k][n]);study_set_host_fields(m->K,zero,zero,zero,zero,zero,Ez);
char path[4096];snprintf(path,sizeof(path),"%s/rank%d_meta.txt",OUTPUT,m->procid);FILE*f=fopen(path,"w");if(!f)exit(9);fprintf(f,"%d %d %d %d %d %d %d %d\n",m->procid,m->nprocs,m->K,p_Np,p_Nfp,p_Nfaces,BSIZE,m->parNtotalout);fclose(f);
dump("initial_Q",m->procid,c_Q,m->K*BSIZE*6*4);dump("initial_rhs",m->procid,c_rhsQ,m->K*BSIZE*6*4);dump("initial_res",m->procid,c_resQ,m->K*BSIZE*6*4);dump("LIFT",m->procid,c_LIFT,p_Np*p_Nfp*p_Nfaces*4);dump("DrDsDt",m->procid,c_DrDsDt,BSIZE*BSIZE*16);dump("vgeo",m->procid,c_vgeo,m->K*12*4);dump("surfinfo",m->procid,c_surfinfo,m->K*p_Nfp*p_Nfaces*7*4);dump("parmapOUT",m->procid,m->c_parmapOUT,m->parNtotalout*4);dump("x",m->procid,m->x[0],valid*8);dump("y",m->procid,m->y[0],valid*8);dump("z",m->procid,m->z[0],valid*8);dump("rk4a_f64",m->procid,m->rk4a,5*8);dump("rk4b_f64",m->procid,m->rk4b,5*8);dump("Npar",m->procid,m->Npar,m->nprocs*4);MPI_Finalize();}
'''
source=B/'cpu_layout.cpp';source.write_text(preamble+body+export)
objects=[];commands=[]
for name in ['Mesh3d','Utils','LoadBalance3d','FacePair3d','ParallelPairs','BuildMaps3d','StartUp3d']:
 o=B/(name+'.o');cmd=['mpicc','-O3','-Dp_N=6','-DNDG3d',*(['-DCUDA']if name=='BuildMaps3d'else[]),'-I'+str(S/'include'),'-c',str(S/'src'/(name+'.c')),'-o',str(o)];subprocess.run(cmd,check=True);commands.append(cmd);objects.append(str(o))
cmd=['mpicxx','-O3','-std=c++17','-Dp_N=6','-DNDG3d','-I'+str(S/'include'),str(source),*objects,'-L'+str(x.parmetis_dir.resolve()),'-lparmetis','-lmetis','-lm','-o',str(B/'cpu_layout')];subprocess.run(cmd,check=True);commands.append(cmd)
(B/'build_provenance.json').write_text(json.dumps(dict(CPU_only=True,no_CUDA_link=True,source_host_packing_reused_not_independent_physics_oracle=True,commands=commands,generated_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n');print(B/'cpu_layout')
