using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function aos_ka!(input,out)
    gid=@index(Global,Linear)-1
    if gid<4096
        value=UInt32(0)
        for k in 0:4095
            value+=reinterpret(UInt32,input[gid*4096+k+1])
        end
        out[gid+1]=reinterpret(Int32,value)
    end
end
@kernel function soa_ka!(input,out)
    gid=@index(Global,Linear)-1
    if gid<4096
        value=UInt32(0)
        for k in 0:4095
            value+=reinterpret(UInt32,input[k*4096+gid+1])
        end
        out[gid+1]=reinterpret(Int32,value)
    end
end
function factory(input,output,layout)
    kernel=layout=="aos" ? aos_ka!(get_backend(input),256) : soa_ka!(get_backend(input),256)
    return ()->kernel(input,output;ndrange=4096)
end
layout_study(factory)
