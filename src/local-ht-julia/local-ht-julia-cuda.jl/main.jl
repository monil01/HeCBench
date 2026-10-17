include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function assembly_ref!(args...)
 index=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 assembly_body!(index÷Int32(32),mod(threadIdx().x-Int32(1),Int32(32)),args...)
 return
end
function capture_ref!(keylen,keypos,reads,hashes,capacity,total)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 capture_hash_body!(i,keylen,keypos,reads,hashes,capacity,total)
 return
end
function run_main!(args...)
 @cuda maxregs=128 threads=512 blocks=1 assembly_ref!(args...)
 nothing
end
function run_hash!(args...)
 @cuda threads=256 blocks=10 capture_ref!(args...)
 nothing
end
study(run_main!,run_hash!)
