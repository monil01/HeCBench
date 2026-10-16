using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function copy_jacc!(global_index,input,output,n::Int64,vec_size::Int32)
    tid=Int64(global_index-1)
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
    spec=JACC.launch_spec(;threads=256,sync=false)
    return ()->JACC.parallel_for(spec,cld(n,Int(vec_size)),copy_jacc!,input,output,n,vec_size)
end
copy_study(factory)
