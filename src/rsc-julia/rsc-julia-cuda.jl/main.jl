include("../study_common.jl")
function rsc_cuda(p,f,n,m,e,conv,id,candidates,counts)
 tx=Int32(threadIdx().x-1);block=Int32(blockIdx().x-1);stride=Int32(gridDim().x);nt=Int32(blockDim().x)
 shared=CUDA.@cuDynamicSharedMem(Int32,1)
 model=block
 while model<m
  if tx==0;shared[1]=0;end
  sync_threads()
  base=4*model
  if p[base+1]!=-2011
   local_count=Int32(0);i=tx
   while i<n
    fv=f[i+1]
    rx=fv.x+unsafe_trunc(Int32,(Float32(fv.x)-p[base+1])*p[base+3])-unsafe_trunc(Int32,(Float32(fv.y)-p[base+2])*p[base+4])-fv.vx
    ry=fv.y+unsafe_trunc(Int32,(Float32(fv.y)-p[base+2])*p[base+3])+unsafe_trunc(Int32,(Float32(fv.x)-p[base+1])*p[base+4])-fv.vy
    local_count+=Int32(abs(rx)>=e||abs(ry)>=e);i+=nt
   end
   CUDA.atomic_add!(pointer(shared,1),local_count);sync_threads()
   if tx==0&&shared[1]<Float32(n)*conv
    j=CUDA.atomic_add!(pointer(id,1),Int32(1))+1;candidates[j]=model;counts[j]=shared[1]
   end
  end
  model+=stride
 end
 return
end
launch(c)=@cuda threads=c.threads blocks=c.blocks shmem=4 rsc_cuda(c.p.data,c.gflows,c.n,c.m,c.e,c.conv,c.id.data,c.c.data,c.o.data)
rsc_study(launch,"cuda_jl")
