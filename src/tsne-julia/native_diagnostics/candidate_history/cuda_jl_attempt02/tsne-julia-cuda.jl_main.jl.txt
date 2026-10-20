using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
const MODEL="cuda_jl"
function tsne_entry!(tag,a,c)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    tsne_body!(tag,i,a,c)
    return
end
function launch!(tag,n,a,c)
    CUDA.@cuda threads=256 blocks=cld(n,256) fastmath=true tsne_entry!(tag,a,c)
    nothing
end
main()
