using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function viterbi_kernel!(old,state,emit,obs,new,path,n::Int32,t::Int32)
    dest=Int32(@index(Global,Linear))-Int32(1)
    if dest<n
        best=0f0;which=Int32(-1);prev=Int32(0)
        while prev<n
            @inbounds score=old[prev+Int32(1)]+state[dest*n+prev+Int32(1)]
            if score>best;best=score;which=prev;end
            prev+=Int32(1)
        end
        @inbounds new[dest+Int32(1)]=best+emit[obs[t+Int32(1)]*n+dest+Int32(1)]
        @inbounds path[(t-Int32(1))*n+dest+Int32(1)]=which
    end
    nothing
end
function study_prepare(old,state,emit,obs,new,path,n)
    k=viterbi_kernel!(CUDABackend(),256)
    t->k(old,state,emit,obs,new,path,Int32(n),Int32(t);ndrange=n)
end
exit(study_main(study_prepare))
