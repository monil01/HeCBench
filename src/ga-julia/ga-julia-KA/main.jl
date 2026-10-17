using CUDA,KernelAbstractions
@kernel function ga_ka!(target,query,result,len,ql,cl,threshold,pos)
    tid=@index(Global,Linear)-1
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
end
function launch_ka!(t,q,o,n,ql,cl,th,pos)
    ga_ka!(get_backend(o),256)(t,q,o,n,ql,cl,th,pos;ndrange=Int(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"));ga_study(launch_ka!)
