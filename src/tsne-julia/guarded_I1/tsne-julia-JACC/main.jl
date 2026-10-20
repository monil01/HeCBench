using CUDA, JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const MODEL="jacc"
const TSNE_JACC_SPECS=Dict{Int,Any}()
function tsne_jacc_callback!(index,tag,a,c)
    tsne_body!(tag,index,a,c)
end
function launch!(tag,n,a,c)
    blocks=cld(n,256)
    spec=get!(TSNE_JACC_SPECS,blocks)do
        JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false,stream=CUDA.legacy_stream())
    end
    JACC.parallel_for(spec,n,tsne_jacc_callback!,tag,a,c)
    nothing
end
println("STUDY_JACC_ENV runtime=",CUDA.runtime_version()," CUDA=",pkgversion(CUDA)," JACC=",pkgversion(JACC))
main()
