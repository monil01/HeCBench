using CUDA
@inline sync_merge()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function diagonal(A,B,path)
 s=CUDA.@cuStaticSharedMem(Int32,37);merge_diagonal_body(Int32(threadIdx().x-1),Int32(blockIdx().x-1),A,B,path,s);return
end
function window(A,B,path,C)
 raw=CUDA.@cuStaticSharedMem(UInt32,1028*(sizeof(eltype(A))÷4)+4);merge_window_body(Int32(threadIdx().x-1),Int32(blockIdx().x-1),A,B,path,C,raw);return
end
function factory(a,b,path,out)
 ()->begin;@cuda threads=32 blocks=112 diagonal(a,b,path);@cuda threads=128 blocks=112 window(a,b,path,out);CUDA.synchronize();end
end
if abspath(PROGRAM_FILE)==@__FILE__;merge_study(factory);end
