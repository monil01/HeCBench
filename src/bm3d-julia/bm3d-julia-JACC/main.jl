using CUDA,JACC,Atomix
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
@inline function model_atomic!(a,i,v)
 Atomix.@atomic a[i] += v
end
function match!(i,a,b,c)
 match_lane!(i,a,b,c)
 nothing
end
function gather!(i,a,b,c,d)
 gather_lane!(i,a,b,c,d)
 nothing
end
function transform!(i,a,inverse)
 transform_lane!(i,a,inverse)
 nothing
end
function hard!(i,a,b,c)
 hard_lane!(i,a,b,c)
 nothing
end
function aggregate!(i,a,b,c,d,e,f,g)
 aggregate_lane!(i,a,b,c,d,e,f,g)
 nothing
end
function final!(i,a,b,c)
 final_lane!(i,a,b,c)
 nothing
end
function factory(a)
 small=JACC.launch_spec(;threads=64,shmem_size=0,sync=false);large=JACC.launch_spec(;threads=128,shmem_size=0,sync=false)
 ()->begin
  JACC.parallel_for(small,768,match!,a.img,a.stacks,a.counts)
  JACC.parallel_for(large,49152,gather!,a.img,a.stacks,a.counts,a.group)
  JACC.parallel_for(large,12288,transform!,a.group,false)
  JACC.parallel_for(small,768,hard!,a.group,a.counts,a.weights)
  JACC.parallel_for(large,12288,transform!,a.group,true)
  JACC.parallel_for(large,49152,aggregate!,a.group,a.stacks,a.counts,a.weights,a.win,a.num,a.den)
  JACC.parallel_for(large,4480,final!,a.num,a.den,a.out)
 end
end
bm_study(factory)
