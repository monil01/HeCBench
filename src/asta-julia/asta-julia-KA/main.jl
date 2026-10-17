using CUDA, KernelAbstractions
@inline block_barrier()=KernelAbstractions.@synchronize()
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function kernel!(A,B,b,input,finished,head)
 s=@localmem(Int32,(2,));cycle_body!(Int32(@index(Local,Linear))-Int32(1),A,B,b,input,finished,head,s)
end
run!(A,B,b,input,finished,head,tmp)=(kernel!(get_backend(input),64)(A,B,b,input,finished,head;ndrange=1024);nothing)
study(run!)
