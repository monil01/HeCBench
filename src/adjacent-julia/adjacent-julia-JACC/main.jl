using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function adjacent_jacc!(index,input,output,left)
    tid=mod(index-1,256);group=div(index-1,256);start=group*1024
    # Load the whole block before any writes, including the in-place right stage.
    tile=JACC.shared(@view input[start+1:start+1024])
    base=tid*4
    for j in 0:3
        localindex=base+j
        value=@inbounds tile[localindex+1]
        neighbor=left ? localindex-1 : localindex+1
        result=(neighbor<0||neighbor>=1024) ? value : value-(@inbounds tile[neighbor+1])
        @inbounds output[start+localindex+1]=result
    end
end
function factory(input,output)
    spec=JACC.launch_spec(;threads=256,shmem_size=4096,sync=false)
    return ()->begin
        JACC.parallel_for(spec,1024*256,adjacent_jacc!,input,output,true)
        JACC.parallel_for(spec,1024*256,adjacent_jacc!,output,output,false)
    end
end
adjacent_study(factory)
