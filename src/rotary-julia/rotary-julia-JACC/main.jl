using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_rotary!(i,o1,o2,x1,x2,cosv,sinv,n::Int64)
    j=Int64(i)-Int64(1);block_start=(j÷Int64(128))*Int64(512);lane=j%Int64(128)
    @inbounds for work in Int64(0):Int64(3)
        idx0=block_start+lane+work*Int64(128)
        if idx0<n
            idx=idx0+Int64(1);a=x1[idx];b=x2[idx];c=cosv[idx];s=sinv[idx];o1[idx]=a*c-b*s;o2[idx]=a*s+b*c
        end
    end
    nothing
end
function study_prepare(o1,o2,a,b,c,s,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=128,blocks=cld(n,512),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_rotary!,spec,cld(n,512)*128,o1,o2,a,b,c,s,Int64(n))
end
exit(study_main(study_prepare))
