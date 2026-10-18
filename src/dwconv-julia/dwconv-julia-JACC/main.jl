using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function portable_dwconv!(i,input,output,weight,bias,args)
    dwconv_lane!(Int32(i-1),input,output,weight,bias,args...)
    nothing
end
function factory(input,output,weight,bias,args)
    n=Int(args[1])
    spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(spec,n,portable_dwconv!,input,output,weight,bias,args)
end
dwconv_study(factory)
