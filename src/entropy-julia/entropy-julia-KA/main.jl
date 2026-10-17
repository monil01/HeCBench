using KernelAbstractions,CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function entropy_ka!(out,input,table,w,h)
 x,y=@index(Global,NTuple);lx,ly=@index(Local,NTuple)
 counts=@localmem Int32 (4096,)
 if x<=w && y<=h
  entropy_point!(out,input,table,counts,x,y,(ly-1)*16+lx,w,h)
 end
end
function entropy_launch!(o,a,t,w,h)
 entropy_ka!(CUDABackend(),(16,16))(o,a,t,w,h;ndrange=(cld(w,16)*16,cld(h,16)*16))
end
study_main(entropy_launch!)
