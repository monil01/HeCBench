using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function pc_jacc!(i,data,out,lut,n::Int32,which::Int32)
    if i<=n
        @inbounds out[(which-Int32(1))*(n+Int32(2))+i+1]=pc_value(data[i],which,lut)
    end
    return
end
function factory(data,out,lut,n)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return which->JACC.parallel_for(spec,n,pc_jacc!,data,out,lut,Int32(n),which)
end
popcount_study(factory)
