using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(x,p,g,gn,w,m,q,out,::Type{T}) where T
 if x<=HH
  tmp,sv,idx=study_value(T,p,g,gn,w,m,q,x)
  @inbounds for c in Int32(0):Int32(WW-1)
   off=c*Int32(HH)+x;z=Float32(study_exp(p[off])*sv);out[off]=T(c==idx ? Float32(tmp)-z : -z)
  end
 end
end
run!(args...)=JACC.parallel_for(JACC.launch_spec(threads=64,blocks=cld(HH,64),shmem_size=0,sync=false),HH,kernel!,args...)
study(run!)
