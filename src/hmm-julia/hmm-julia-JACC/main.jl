using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function viterbi_kernel!(index,old,state,emit,obs,new,path,n::Int32,t::Int32)
    dest=Int32(index)-Int32(1)
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
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    t->JACC.parallel_for(viterbi_kernel!,spec,n,old,state,emit,obs,new,path,Int32(n),Int32(t))
end
exit(study_main(study_prepare))
