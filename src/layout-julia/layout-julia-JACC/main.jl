using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function aos_jacc!(index,input,out)
    gid=index-1
    if gid<4096
        value=UInt32(0)
        for k in 0:4095
            value+=reinterpret(UInt32,input[gid*4096+k+1])
        end
        out[gid+1]=reinterpret(Int32,value)
    end
end
function soa_jacc!(index,input,out)
    gid=index-1
    if gid<4096
        value=UInt32(0)
        for k in 0:4095
            value+=reinterpret(UInt32,input[k*4096+gid+1])
        end
        out[gid+1]=reinterpret(Int32,value)
    end
end
function factory(input,output,layout)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    kernel=layout=="aos" ? aos_jacc! : soa_jacc!
    return ()->JACC.parallel_for(spec,4096,kernel,input,output)
end
layout_study(factory)
