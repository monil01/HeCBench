using CUDA,JACC
include(joinpath(@__DIR__,"..","study_math.jl"))
function multinomial_jacc!(i,dest,nd,nc,samples,values,sd,sc)
    if (i-1)%256==0;multinomial_row!(Int32((i-1)÷256),dest,nd,nc,samples,values,sd,sc);end
    return nothing
end
const SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(args...)
    JACC.parallel_for(SPEC,512*256,multinomial_jacc!,args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"));multinomial_study(launch_jacc!)
