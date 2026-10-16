using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function copy_ka!(input,output,n::Int64,vec_size::Int32)
    tid=Int64(@index(Global,Linear)-1)
    index=tid*Int64(vec_size)+1
    j=Int32(0)
    while j<vec_size
        pos=index+Int64(j)
        if pos<=n
            output[pos]=input[pos]
        end
        j+=Int32(1)
    end
end
function factory(input,output,n,vec_size)
    kernel=copy_ka!(get_backend(input),256)
    return ()->kernel(input,output,n,vec_size;ndrange=cld(n,Int(vec_size)))
end
copy_study(factory)
