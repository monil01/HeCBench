using JACC,CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
const ENTROPY_TEMPLATE=CUDA.zeros(Int32,4096)
const ENTROPY_BACKEND=JACC.get_backend(:cuda)
const ENTROPY_SPEC=JACC.LaunchSpec{typeof(ENTROPY_BACKEND)}(;threads=(16,16),blocks=(cld(EW,16),cld(EH,16)),shmem_size=16384,sync=false)
function entropy_jacc!(x,y,out,input,table,template,w,h)
 counts=JACC.shared(template)
 if x<=w && y<=h
  lx=(x-1)%16+1;ly=(y-1)%16+1
  entropy_point!(out,input,table,counts,x,y,(ly-1)*16+lx,w,h)
 end
 nothing
end
function entropy_launch!(o,a,t,w,h)
 # Padded domain ensures every thread participates in JACC.shared synchronization.
 JACC.parallel_for(entropy_jacc!,ENTROPY_SPEC,(cld(w,16)*16,cld(h,16)*16),o,a,t,ENTROPY_TEMPLATE,w,h)
end
study_main(entropy_launch!)
