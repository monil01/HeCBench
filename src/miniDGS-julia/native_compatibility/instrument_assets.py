from pathlib import Path
import argparse,re
p=Path(__file__).resolve();a=argparse.ArgumentParser();a.add_argument('--source-copy',type=Path,required=True);x=a.parse_args();P=x.source_copy
kernel=P/'src/MaxwellsKernel3d.cu';s=kernel.read_text()
helper=r'''
static void study_dump(const char*name,int rank,const void*data,size_t bytes){
 char path[4096];snprintf(path,sizeof(path),"%s/rank%d_%s.bin",getenv("STUDY_NATIVE_CAPTURE_DIR"),rank,name);
 FILE*f=fopen(path,"wb");if(!f||fwrite(data,1,bytes,f)!=bytes){fprintf(stderr,"capturewritefailure%s\n",path);exit(9);}fclose(f);
}
static void study_dump_gpu(const char*name,int rank,const void*data,size_t bytes){
 void*host=malloc(bytes);cudaError_t e=cudaMemcpy(host,data,bytes,cudaMemcpyDeviceToHost);if(e!=cudaSuccess){fprintf(stderr,"captureCUDA%d:%s\n",int(e),cudaGetErrorString(e));exit(10);}study_dump(name,rank,host,bytes);free(host);
}
void study_export_assets(Mesh*mesh){
 char path[4096];snprintf(path,sizeof(path),"%s/rank%d_meta.txt",getenv("STUDY_NATIVE_CAPTURE_DIR"),mesh->procid);FILE*f=fopen(path,"w");if(!f)exit(11);
 fprintf(f,"%d %d %d %d %d %d %d %d\n",mesh->procid,mesh->nprocs,mesh->K,p_Np,p_Nfp,p_Nfaces,BSIZE,mesh->parNtotalout);fclose(f);
 study_dump_gpu("initial_Q",mesh->procid,c_Q,mesh->K*BSIZE*p_Nfields*4);
 study_dump_gpu("initial_rhs",mesh->procid,c_rhsQ,mesh->K*BSIZE*p_Nfields*4);
 study_dump_gpu("initial_res",mesh->procid,c_resQ,mesh->K*BSIZE*p_Nfields*4);
 study_dump_gpu("LIFT",mesh->procid,c_LIFT,p_Np*p_Nfp*p_Nfaces*4);
 study_dump_gpu("DrDsDt",mesh->procid,c_DrDsDt,BSIZE*BSIZE*16);
 study_dump_gpu("vgeo",mesh->procid,c_vgeo,mesh->K*12*4);
 study_dump_gpu("surfinfo",mesh->procid,c_surfinfo,mesh->K*p_Nfp*p_Nfaces*7*4);
 study_dump_gpu("parmapOUT",mesh->procid,mesh->c_parmapOUT,mesh->parNtotalout*4);
 study_dump("xyz",mesh->procid,mesh->x[0],0); // Separatecoordinatearrays follow; thezero-bytefile is explicit, not an oracle.
 study_dump("x",mesh->procid,mesh->x[0],mesh->K*p_Np*8);
 study_dump("y",mesh->procid,mesh->y[0],mesh->K*p_Np*8);
 study_dump("z",mesh->procid,mesh->z[0],mesh->K*p_Np*8);
 study_dump("rk4a_f64",mesh->procid,mesh->rk4a,5*8);
 study_dump("rk4b_f64",mesh->procid,mesh->rk4b,5*8);
 study_dump("Npar",mesh->procid,mesh->Npar,mesh->nprocs*4);
}
static void study_stage_gpu(const char*part,int stage,Mesh*mesh,const void*data,size_t bytes){char name[256];snprintf(name,sizeof(name),"stage%d_%s",stage,part);study_dump_gpu(name,mesh->procid,data,bytes);}
static void study_stage_host(const char*part,int stage,Mesh*mesh,const void*data,size_t bytes){char name[256];snprintf(name,sizeof(name),"stage%d_%s",stage,part);study_dump(name,mesh->procid,data,bytes);}
'''
anchor='/* assumes data resides on device */';assert s.count(anchor)==1;s=s.replace(anchor,helper+'\n'+anchor)
s=s.replace('void MaxwellsKernel3d(Mesh *mesh, float frka, float frkb, float fdt){','void MaxwellsKernel3d(Mesh *mesh, float frka, float frkb, float fdt){\nstatic int study_stage=0;study_stage++;')
s=s.replace('MaxwellsMPISend3d(mesh);','MaxwellsMPISend3d(mesh);\nstudy_stage_host("gathered",study_stage,mesh,mesh->f_outQ,mesh->parNtotalout*4);')
s=s.replace('MaxwellsGPU_VOL_Kernel3D <<< BlocksPerGrid, ThreadsPerBlock >>>  (c_rhsQ);','MaxwellsGPU_VOL_Kernel3D <<< BlocksPerGrid, ThreadsPerBlock >>>  (c_rhsQ);\nstudy_stage_gpu("volume_rhs",study_stage,mesh,c_rhsQ,mesh->K*BSIZE*p_Nfields*4);')
s=s.replace('MaxwellsMPIRecv3d(mesh, c_partQ);','MaxwellsMPIRecv3d(mesh, c_partQ);\nstudy_stage_gpu("received",study_stage,mesh,c_partQ,mesh->parNtotalout*4);')
s=s.replace('MaxwellsGPU_SURF_Kernel3D <<< BlocksPerGrid, ThreadsPerBlock >>> (c_rhsQ);','MaxwellsGPU_SURF_Kernel3D <<< BlocksPerGrid, ThreadsPerBlock >>> (c_rhsQ);\nstudy_stage_gpu("surface_rhs",study_stage,mesh,c_rhsQ,mesh->K*BSIZE*p_Nfields*4);')
s=s.replace('(Ntotal, c_resQ, c_rhsQ, c_Q, frka, frkb, fdt);','(Ntotal, c_resQ, c_rhsQ, c_Q, frka, frkb, fdt);\nstudy_stage_gpu("res",study_stage,mesh,c_resQ,Ntotal*4);study_stage_gpu("Q",study_stage,mesh,c_Q,Ntotal*4);')
kernel.write_text(s)
driver=P/'src/MaxwellsDriver3d.c';s=driver.read_text();s=s.replace('gpu_set_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);','gpu_set_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);\n  extern void study_export_assets(Mesh*);study_export_assets(mesh);');driver.write_text(s)
print('instrumented untimeddiagnostic',P)
