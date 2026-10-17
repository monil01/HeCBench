using CUDA
@inline block_barrier()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function scan!(lo,hi,input,num)
 tid=threadIdx().x-Int32(1);bid=blockIdx().x-Int32(1);s=CuStaticSharedArray(Int32,512);scan_body!(tid,bid,gridDim().x,lo,hi,input,num,s);return
end
function finalize!(lo,hi,bits,minimum,num)
 s=CuStaticSharedArray(Int32,512);finalize_body!(threadIdx().x-Int32(1),lo,hi,bits,minimum,num,s);return
end
function pack!(bits,minimum,out,input,num)
 s=CuStaticSharedArray(UInt32,256);pack_body!(threadIdx().x-Int32(1),blockIdx().x-Int32(1),gridDim().x,bits,minimum,out,input,num,s);return
end
function run!(input,num,lo,hi,bits,minimum,out,tmp,packtmp)
 @cuda threads=256 blocks=length(lo) scan!(lo,hi,input,num)
 @cuda threads=256 blocks=1 finalize!(lo,hi,bits,minimum,num)
 @cuda threads=256 blocks=length(lo) pack!(bits,minimum,out,input,num)
 return
end
study(run!)
