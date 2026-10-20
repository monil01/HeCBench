from pathlib import Path
p=Path(__file__).resolve().parents[1];s='using CUDA\ninclude(joinpath(@__DIR__,"..","device_primitives.jl"))\n'
for name,args,nshared,template in [('match','input,fs,ds,tf,td',5188,'match'),('scan','input,out,sums,n',2048,'scan'),('pack','fo,doo,tf,td,pf,pd',0,'pack'),('decode','out,fo,doo,pf,pd,nb',0,'decode')]:
 s+=f'function {name}_kernel({args})\n tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid\n'
 if nshared:s+=f' s=CUDA.@cuStaticSharedMem(UInt8,{nshared})\n'
 s+=' @inbounds begin\n'+(p/'translation_generators'/f'{template}_body.jl.txt').read_text().replace('SYNC','CUDA.sync_threads()')+'\nend\nreturn nothing\nend\n'
s+='''function add_kernel(out,offsets,n)
 i=(Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)-1
 if i<n;@inbounds out[i+1]+=offsets[div(i,512)+1];end
 return nothing
end
alloc_backend(::Type{T},n) where T=CUDA.CuArray{T}(undef,n)
function run_match!(input,fs,ds,tf,td,nb);CUDA.@cuda threads=128 blocks=nb match_kernel(input,fs,ds,tf,td);end
function run_scan!(input,out,sums,n,groups);CUDA.@cuda threads=128 blocks=groups scan_kernel(input,out,sums,n);end
function run_add!(out,offsets,n);CUDA.@cuda threads=128 blocks=cld(n,128) add_kernel(out,offsets,n);end
function run_pack!(fo,doo,tf,td,pf,pd,nb);CUDA.@cuda threads=128 blocks=nb pack_kernel(fo,doo,tf,td,pf,pd);end
function run_decode!(out,fo,doo,pf,pd,nb);CUDA.@cuda threads=32 blocks=cld(nb,32) decode_kernel(out,fo,doo,pf,pd,nb);end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
'''
(p/'lzss-julia-cuda.jl'/'main.jl').write_text(s)
