using CUDA, KernelAbstractions
include("/tmp/hecbench-roofline-v100-20261020/batch069/tsne/ka/scratch/profile_source/study_common.jl")
const MODEL="ka"
@kernel function tsne_ka_entry!(tag,a,c)
    i=@index(Global,Linear)
    tsne_body!(tag,i,a,c)
end
function launch!(tag,n,a,c)
    tsne_ka_entry!(CUDABackend(),256)(tag,a,c;ndrange=n)
    nothing
end
main()
