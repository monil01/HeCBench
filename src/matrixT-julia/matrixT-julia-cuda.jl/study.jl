include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_prepare(out,input,N,kind)
    kernels=(copy_kernel!,copy_shared_kernel!,transpose_coarse_kernel!,transpose_fine_kernel!,transpose_naive_kernel!,transpose_coalesced_kernel!,transpose_pad_kernel!,transpose_swizzle_kernel!,transpose_diagonal_kernel!);k=kernels[kind+1]
    ()->(@cuda threads=(16,16) blocks=(N÷16,N÷16) k(out,input,Int32(N),Int32(N)))
end
exit(study_main(study_prepare))
