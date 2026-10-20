"""Reproduce minimal launch/allocation hooks; preserve scientific tree/kernel body."""
from pathlib import Path
import shutil,re,json,hashlib
B=Path(__file__).resolve().parent;C=B/'configured_native';C.mkdir(exist_ok=True)
for p in (B/'reference_history').iterdir():
 if p.is_file() and p.suffix in ['.cxx','.h','.c','.mk']and p.name!='dims-local.c':shutil.copy2(p,C/p.name)
p=B/'reference_history/RCBForceTree.cxx';s=p.read_text();s=s.replace('#include "cudaUtil.h"','#include "cudaUtil.h"\n#include "../halo_dispatch.h"\nstd::map<void*,HaloAllocation>halo_allocations;\nHaloCallback halo_target_callback=nullptr;\nHaloObserver halo_observer_callback=nullptr;\nextern "C" void halo_set_callbacks(HaloCallback target,HaloObserver observer){halo_target_callback=target;halo_observer_callback=observer;}\nextern "C" size_t halo_argument_size(){return sizeof(HaloArgs);}')
s=s.replace('cudaMallocHost(','halo_malloc_host(').replace('cudaFreeHost(','halo_free_host(')
pattern=r'  Step10_kernel <<< dim3\(blocks\), dim3\(threads\), 0, stream >>> \(\s*count,count1,xx,yy,zz,mass,xx1,yy1,zz1,mass1, vx, vy, vz, fsrrmax2, rsm2, fcoeff\);'
new='''  HaloArgs args{int(count),int(count1),int(blocks.x),int(blocks.y),int(threads.x),int(threads.y),xx,yy,zz,mass,xx1,yy1,zz1,mass1,vx,vy,vz,fsrrmax2,rsm2,fcoeff,(void*)stream};
  if(halo_observer_callback)halo_observer_callback(0,&args);
  if(halo_target_callback){
    if(halo_target_callback(&args)!=0){fprintf(stderr,"HALO_TARGET_CALLBACK_REJECTED\\n");exit(9);}
  }else{
    Step10_kernel <<< dim3(blocks), dim3(threads), 0, stream >>> (
      count,count1,xx,yy,zz,mass,xx1,yy1,zz1,mass1, vx, vy, vz, fsrrmax2, rsm2, fcoeff);
  }
  if(halo_observer_callback){cudaDeviceSynchronize();halo_observer_callback(1,&args);}
'''
s,n=re.subn(pattern,lambda m:new,s);assert n==1
(C/'RCBForceTree.cxx').write_text(s)
raw=p.read_text();start=raw.index('typedef long long int int64;');end=raw.index('\n#endif\n\n#ifdef __bgq__',start);assert raw[start:end]in s
x={'original_RCB_source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'configured_RCB_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'original_GPU_body_bytes_unchanged':True,'original_CPU_tree/list_functions_unchanged':True,'changes':['One scientific GPU launch wrapped in target/observer dispatch; native branch originalkernelcall unchanged.','Pinnedhostallocation guard/zero wrappers preserve cudaMallocHost memorytype/sourcepayloadinitialzeros andalignments.','Targetpostdispatch wholecontextcompletion covers source/Julia differingCUDAstreams; observations explicitlyoutsideROI.'],'target_native_GPU_fallback_forbidden_by_nonnullcallback':True};(B/'tree_adapter_provenance.json').write_text(json.dumps(x,indent=2)+'\n');print(x)
