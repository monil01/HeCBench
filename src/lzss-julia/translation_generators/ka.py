from pathlib import Path
p=Path(__file__).resolve().parents[1];s='using CUDA, KernelAbstractions\ninclude(joinpath(@__DIR__,"..","device_primitives.jl"))\n'
for name,args,nshared,template in [('match','input,fs,ds,tf,td',5188,'match'),('scan','input,out,sums,n',2048,'scan'),('pack','fo,doo,tf,td,pf,pd',0,'pack'),('decode','out,fo,doo,pf,pd,nb',0,'decode')]:
 s+=f'@kernel function {name}_kernel({args})\n tid=Int(@index(Local,Linear))-1;gid=Int(@index(Group,Linear))-1;i=Int(@index(Global,Linear))-1\n'
 if nshared:s+=f' s=@localmem UInt8 ({nshared},)\n'
 s+=' @inbounds begin\n'+(p/'translation_generators'/f'{template}_body.jl.txt').read_text().replace('SYNC','@synchronize()')+'\nend\nend\n'
s+='''@kernel function add_kernel(out,offsets,n)
 i=Int(@index(Global,Linear))-1
 if i<n;@inbounds out[i+1]+=offsets[div(i,512)+1];end
end
alloc_backend(::Type{T},n) where T=KernelAbstractions.allocate(CUDABackend(),T,(n,))
function run_match!(input,fs,ds,tf,td,nb);match_kernel(CUDABackend(),128)(input,fs,ds,tf,td;ndrange=nb*128);end
function run_scan!(input,out,sums,n,groups);scan_kernel(CUDABackend(),128)(input,out,sums,n;ndrange=groups*128);end
function run_add!(out,offsets,n);add_kernel(CUDABackend(),128)(out,offsets,n;ndrange=cld(n,128)*128);end
function run_pack!(fo,doo,tf,td,pf,pd,nb);pack_kernel(CUDABackend(),128)(fo,doo,tf,td,pf,pd;ndrange=nb*128);end
function run_decode!(out,fo,doo,pf,pd,nb);decode_kernel(CUDABackend(),32)(out,fo,doo,pf,pd,nb;ndrange=cld(nb,32)*32);end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
'''
(p/'lzss-julia-KA'/'main.jl').write_text(s)
