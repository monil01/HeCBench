using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel cpu=false function assembly_ka!(ctg_offsets,contigs,reads,quals,read_offsets,read_counts,prefix_ht,depth,keylen,keypos,counts,exts,extcounts,visited_len,visited_pos,visited_val,longest,temporary,walklens,kstart,maxwalk,maxmer,ncontigs,qualoffset)
 index=Int32(@index(Global,Linear))-Int32(1);lane=mod(Int32(@index(Local,Linear))-Int32(1),Int32(32))
 assembly_body!(index÷Int32(32),lane,ctg_offsets,contigs,reads,quals,read_offsets,read_counts,prefix_ht,depth,keylen,keypos,counts,exts,extcounts,visited_len,visited_pos,visited_val,longest,temporary,walklens,kstart,maxwalk,maxmer,ncontigs,qualoffset)
end
@kernel cpu=false function capture_ka!(keylen,keypos,reads,hashes,capacity,total)
 i=Int32(@index(Global,Linear))-Int32(1)
 capture_hash_body!(i,keylen,keypos,reads,hashes,capacity,total)
end
const BACKEND=CUDABackend()
function run_main!(args...)
 assembly_ka!(BACKEND,512)(args...;ndrange=512)
 nothing
end
function run_hash!(args...)
 capture_ka!(BACKEND,256)(args...;ndrange=2560)
 nothing
end
study(run_main!,run_hash!)
