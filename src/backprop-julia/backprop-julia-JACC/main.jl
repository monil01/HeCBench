using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const BP_TEMPLATE=CUDA.zeros(Float32,272)
const BP_BACKEND=JACC.get_backend(:cuda)
function layerforward_jacc!(x,gy,input,weights,partial,template)
 y=(gy-1)%16+1;by=(gy-1)÷16;idx=gy*17+x+1;ti=16+(y-1)*16+x
 shared=JACC.shared(template)
 if x==1;shared[y]=input[gy+1];end
 JACC.sync_workgroup()
 shared[ti]=weights[idx]
 JACC.sync_workgroup()
 shared[ti]*=shared[y]
 JACC.sync_workgroup()
 power=2
 while power<=16
  if (y-1)%power==0;shared[ti]+=shared[ti+(power÷2)*16];end
  JACC.sync_workgroup();power*=2
 end
 weights[idx]=shared[ti]
 JACC.sync_workgroup()
 if x==1;partial[by*16+y]=shared[16+y];end
 nothing
end
function adjust_jacc!(x,gy,input,weights,delta,prev)
 y=(gy-1)%16+1;by=(gy-1)÷16;idx=gy*17+x+1
 u=ETA*delta[x+1]*input[gy+1]+MOMENTUM*prev[idx];weights[idx]+=u;prev[idx]=u
 JACC.sync_workgroup()
 if y==1 && by==0
  bi=x+1;v=ETA*delta[bi]+MOMENTUM*prev[bi];weights[bi]+=v;prev[bi]=v
 end
 nothing
end
function forward!(a,w,p,n)
 spec=JACC.LaunchSpec{typeof(BP_BACKEND)}(;threads=(16,16),blocks=(1,n÷16),shmem_size=1088,sync=false)
 JACC.parallel_for(layerforward_jacc!,spec,(16,n),a,w,p,BP_TEMPLATE)
end
function adjust!(a,w,d,p,n)
 spec=JACC.LaunchSpec{typeof(BP_BACKEND)}(;threads=(16,16),blocks=(1,n÷16),shmem_size=0,sync=false)
 JACC.parallel_for(adjust_jacc!,spec,(16,n),a,w,d,p)
end
study_main(net->study_train_tiled!(net,forward!,adjust!))
