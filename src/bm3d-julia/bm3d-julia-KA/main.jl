using CUDA,KernelAbstractions,Atomix
include("../study_device.jl")
include("../study_common.jl")
@inline function model_atomic!(a,i,v)
 Atomix.@atomic a[i] += v
end
@kernel function match!(a,b,c)
 i=@index(Global,Linear)
 match_lane!(i,a,b,c)
end
@kernel function gather!(a,b,c,d)
 i=@index(Global,Linear)
 gather_lane!(i,a,b,c,d)
end
@kernel function transform!(a,inverse)
 i=@index(Global,Linear)
 transform_lane!(i,a,inverse)
end
@kernel function hard!(a,b,c)
 i=@index(Global,Linear)
 hard_lane!(i,a,b,c)
end
@kernel function aggregate!(a,b,c,d,e,f,g)
 i=@index(Global,Linear)
 aggregate_lane!(i,a,b,c,d,e,f,g)
end
@kernel function final!(a,b,c)
 i=@index(Global,Linear)
 final_lane!(i,a,b,c)
end
function factory(a)
 backend=CUDABackend();matchk=match!(backend,64);gatherk=gather!(backend,128);transk=transform!(backend,128);hardk=hard!(backend,64);aggk=aggregate!(backend,128);finalk=final!(backend,128)
 ()->begin
  matchk(a.img,a.stacks,a.counts;ndrange=768)
  gatherk(a.img,a.stacks,a.counts,a.group;ndrange=49152)
  transk(a.group,false;ndrange=12288)
  hardk(a.group,a.counts,a.weights;ndrange=768)
  transk(a.group,true;ndrange=12288)
  aggk(a.group,a.stacks,a.counts,a.weights,a.win,a.num,a.den;ndrange=49152)
  finalk(a.num,a.den,a.out;ndrange=4480)
 end
end
bm_study(factory)
