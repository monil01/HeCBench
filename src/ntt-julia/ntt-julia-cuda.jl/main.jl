include(joinpath(@__DIR__,"..","study_common.jl"))
function ntt!(y,x)
 t=Int32(threadIdx().x)-Int32(1);b=Int32(blockIdx().x)-Int32(1);buffer=CuStaticSharedArray(UInt64,512)
 phase1!(x,buffer,b,t);sync_threads();phase2!(y,buffer,b,t);return
end
run_study((y,x)->(@cuda threads=64 blocks=128 ntt!(y,x)),"cuda_jl")
