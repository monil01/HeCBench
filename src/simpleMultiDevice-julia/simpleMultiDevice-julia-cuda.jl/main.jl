include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA
CUDA.allowscalar(false)
function reduce_device(out,input,n)
 tid=(blockIdx().x-1)*blockDim().x+threadIdx().x
 pos=Int(tid);total=0f0
 while pos<=n;total+=input[pos];pos+=8192;end
 out[tid]=total
 return
end
launch(o,i,n)=(@cuda threads=256 blocks=32 reduce_device(o,i,Int32(n));nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->CUDA.CuArray{Float32}(undef,n),launch)
