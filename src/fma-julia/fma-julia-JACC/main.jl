using CUDA,JACC
function fma_basic_jacc!(i,a,b,c,ins,outs,ops,ch)
    idx=i-1;op=idx÷ch;k=idx%ch
    @inbounds c[outs[op+1]*ch+k+1]+=a[ins[op+1]*ch+k+1]*b[k+1]
    return nothing
end
function fma_row_jacc!(op,a,b,c,ins,outs,ops,ch)
    for k in 0:ch-1
        @inbounds c[outs[op]*ch+k+1]+=a[ins[op]*ch+k+1]*b[k+1]
    end
    return nothing
end
const SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(a,b,c1,c2,ins,outs,ops,ch,na,nc)
    JACC.parallel_for(SPEC,Int(ops*ch),fma_basic_jacc!,a,b,c1,ins,outs,ops,ch)
    JACC.parallel_for(SPEC,Int(ops),fma_row_jacc!,a,b,c2,ins,outs,ops,ch)
end
include(joinpath(@__DIR__,"..","study_common.jl"));fma_study(launch_jacc!)
