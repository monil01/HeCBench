using CUDA,JACC,Printf,SHA
include(joinpath(@__DIR__,"..","columnar_math.jl"))
@inline function setup_site!(i,state)
    idx=Int32(i-1);s=UInt32(idx)
    for j in Int32(0):idx-Int32(1);s=cs_step(s);end
    @inbounds state[idx+1]=s
    nothing
end
@inline function decode_site!(i,scores,encrypted,seeds,output,diag)
    # Whole source676-word read-only vector fits exact2704B dynamic shared region.
    # Source allocation already has actual input readbacks and physical guards.
    shared=JACC.shared(scores)
    cs_decode_site!(Int32(i-1),shared,encrypted,seeds,output,diag)
    nothing
end
function launch_both!(scores,cipher,seeds,out,diag)
    setup=JACC.launch_spec(threads=32,blocks=32,shmem_size=0,sync=false,stream=CUDA.stream())
    decode=JACC.launch_spec(threads=32,blocks=32,shmem_size=2704,sync=false,stream=CUDA.stream())
    JACC.parallel_for(setup,1024,setup_site!,seeds)
    JACC.parallel_for(decode,1024,decode_site!,scores,cipher,seeds,out,diag)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
