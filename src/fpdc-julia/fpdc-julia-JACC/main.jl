using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function compression!(index,dim,src,dst,cuts,offsets)
    tid=Int32(CUDA.threadIdx().x-1);global0=Int32(index-1)
    scratch=CUDA.CuStaticSharedArray(Int32,1536)
    fpdc_lane!(dim,src,dst,cuts,offsets,scratch,tid,global0)
    nothing
end
function factory(src,dst,cuts,offsets)
    spec=JACC.launch_spec(;threads=64,blocks=4,shmem_size=0,sync=false)
    dim->JACC.parallel_for(spec,256,compression!,dim,src,dst,cuts,offsets)
end
fpdc_study(factory)
