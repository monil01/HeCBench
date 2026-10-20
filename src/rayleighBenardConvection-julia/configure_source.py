from pathlib import Path
import re,json,hashlib
P=Path(__file__).resolve().parent;s=(P/'reference_history/main.cu').read_text()
# Preserve native source; derived host integration redirects exactly seven pointwise entry calls.
s=s.replace('int main(int argc, char* argv[]) {','int source_main(int argc, char* argv[]) {')
for id,name in enumerate(['ElemMultOmega','ElemMultT','ElemMultNu','SubOne','AddOne','AddX']):
 pat=rf'{name}<<<([^;]+?)>>>\(([^;]+?)\);'
 def f(m):
  geometry=m[1];args=m[2]; a=[x.strip() for x in args.split(',')]
  n=['(M-2)*(N-2)','(M-2)*N','N','N','N','(M-2)*N'][id]
  bb=a[1] if len(a)>1 else 'nullptr'
  if geometry=='grid, block':g='grid.x,grid.y,block.x'
  else:g='(unsigned)(floorf(N/256.f)+1),1,256'
  return '{ if(study_cb) { RBCall q={'+f'{id},{n},{g},0,0,0,{a[0]},{bb},nullptr'+'}; if(study_cb(&q)) abort(); } else '+m[0]+' }'
 s=re.sub(pat,f,s)
s=s.replace('Updatedt<<<1,1>>>(iu, u, iv, v, dt);','{if(study_cb){RBCall q={6,1,1,1,1,iu,iv,0,u,v,dt};if(study_cb(&q))abort();}else Updatedt<<<1,1>>>(iu,u,iv,v,dt);}')
# Per-step actual timestep is a retained diagnostic, not part of timing by omission: common memcpy was already source CUBLAS GetVector.
s=s.replace('frames += dt;','frames += dt;\n    study_dt_history.push_back(dt);')
s=s.replace('printf("Nusselt number: %.1f\\n", nunum);','study_nu_history.push_back(nunum);')
inputs=[('d_dt','1'),('d_T','N*(M-2)'),('d_psi','N*(M-2)'),('d_dsc','(M-2)*(M-2)'),('d_dsr','(N-2)*(N-2)'),('d_ei','(M-2)*(N-2)'),('d_tr','M*N'),('d_trnu','N'),('d_ztop','4*N'),('d_zbot','4*N'),('d_nutop','4*N'),('d_nubot','4*N'),('d_u','N*(M-2)'),('d_v','N*(M-2)'),('d_xrk3','N*(M-2)'),('d_yrk3','N*(M-2)'),('d_zrk3','N*(M-2)'),('d_y','N*(M-2)'),('d_Tbuff','N*(M-2)'),('d_DxT','N*(M-2)'),('d_temp','N*(M-2)')]
outputs=[('d_T','N*(M-2)'),('d_dt','1'),('d_psi','N*(M-2)'),('d_omega','(M-2)*(N-2)'),('d_u','N*(M-2)'),('d_v','N*(M-2)'),('d_xrk3','N*(M-2)'),('d_yrk3','N*(M-2)'),('d_zrk3','N*(M-2)'),('d_y','N*(M-2)'),('d_Tbuff','N*(M-2)'),('d_DxT','N*(M-2)'),('d_temp','N*(M-2)'),('d_nutop','4*N'),('d_nubot','4*N')]
s=s.replace('auto start = std::chrono::steady_clock::now();','study_dt_history.clear();study_nu_history.clear();study_inputs.clear();study_outputs.clear();\n'+ '\n'.join(f'study_read({p},{n},study_inputs);' for p,n in inputs)+'\n CUDA_CHECK(cudaDeviceSynchronize());\n auto start = std::chrono::steady_clock::now();')
s=s.replace('auto end = std::chrono::steady_clock::now();','CUDA_CHECK(cudaDeviceSynchronize());\n auto end = std::chrono::steady_clock::now();')
s=s.replace('printf("Total compute time: %f (s)\\n", time_sec);','study_seconds=std::chrono::duration<double>(end-start).count();\n'+ '\n'.join(f'study_read({p},{n},study_outputs);' for p,n in outputs)+'\n study_outputs.insert(study_outputs.end(),study_dt_history.begin(),study_dt_history.end()); study_outputs.insert(study_outputs.end(),study_nu_history.begin(),study_nu_history.end()); study_outputs.push_back(frames); study_outputs.push_back(float(tstep));\n printf("Total compute time: %f (s)\\n", time_sec);')
header='''#include <vector>\n#include <fstream>\n#include <string>\n#include <cmath>\n#include <cuda_runtime.h>\nstruct RBCall { int op,n,gx,gy,threads,iu,iv,pad;float *a,*b,*out; };\nstatic_assert(sizeof(RBCall)==56,"ABI");\nusing Callback=int(*)(RBCall*);static Callback study_cb=nullptr;\nstatic std::vector<float> study_inputs,study_outputs,study_dt_history,study_nu_history;static double study_seconds;\n#define CUDA_CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\\n",cudaGetErrorString(e));abort();}}while(0)\nstatic void study_read(const float*p,size_t n,std::vector<float>&v){size_t off=v.size();v.resize(off+n);CUDA_CHECK(cudaMemcpy(v.data()+off,p,n*sizeof(float),cudaMemcpyDeviceToHost));}\n'''
s=s.replace('(unsigned)(floorf(N/256.f)+1)','(int)(floorf(N/256.f)+1)').replace('grid.x,grid.y,block.x','int(grid.x),int(grid.y),int(block.x)')
(P/'configured_source.cuh').write_text(header+s)
(P/'layout.json').write_text(json.dumps({'M':500,'N':1000,'steps':1000,'inputs':inputs,'outputs':outputs,'dynamic_output_tail':'1000 actual dt, variable Nu samples, frames, tstep','schema':'float32 native little endian'},indent=2)+'\n')
(P/'source_archive.json').write_text(json.dumps({'original_files':[{ 'path':str(p.relative_to(P)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in (P/'reference_history').iterdir()], 'modified_native_gpu_bodies':False,'controller_integration':'Seven same-signature dispatch hooks; source1000step full500x1000 shape unchanged; common input/output D2H outside ROI, device-wide completion, Nuprintf replaced inROI same-model vector capture.'},indent=2)+'\n')
