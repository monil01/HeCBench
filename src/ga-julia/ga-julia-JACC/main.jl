using CUDA,JACC
function ga_jacc!(i,target,query,result,len,ql,cl,threshold,pos)
    tid=i-1
    if tid<len
        matched=false
        for i in 0:ql-cl
            distance=Int32(0)
            for j in 0:cl-1
                @inbounds distance+=Int32(target[pos+tid+j+1]!=query[i+j+1])
            end
            if distance<threshold;matched=true;break;end
        end
        if matched;@inbounds result[tid+1]=0x01;end
    end
    return nothing
end
const SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(t,q,o,n,ql,cl,th,pos)
    JACC.parallel_for(SPEC,Int(n),ga_jacc!,t,q,o,n,ql,cl,th,pos)
end
include(joinpath(@__DIR__,"..","study_common.jl"));ga_study(launch_jacc!)
