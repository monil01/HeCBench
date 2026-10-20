from pathlib import Path
p=Path(__file__).resolve().parents[1];s='using CUDA\nimport JACC\nJACC.@init_backend\ninclude(joinpath(@__DIR__,"..","device_primitives.jl"))\n'
for name,args,nshared,template in [('match','input,fs,ds,tf,td',5188,'match'),('scan','input,out,sums,n',2048,'scan'),('pack','fo,doo,tf,td,pf,pd',0,'pack'),('decode','out,fo,doo,pf,pd,nb',0,'decode')]:
 nargs=args+(',template'if nshared else'');threads=32 if name=='decode' else 128
 s+=f'function {name}_kernel(index,{nargs})\n i=Int(index)-1;tid=i%{threads};gid=div(i,{threads})\n'
 if nshared:s+=' s=JACC.shared(template)\n'
 s+=' @inbounds begin\n'+(p/'translation_generators'/f'{template}_body.jl.txt').read_text().replace('SYNC','JACC.sync_workgroup()')+'\nend\nreturn nothing\nend\n'
s+='''function add_kernel(index,out,offsets,n)
 i=Int(index)-1
 if i<n;@inbounds out[i+1]+=offsets[div(i,512)+1];end
 return nothing
end
alloc_backend(::Type{T},n) where T=JACC.array(T,n)
const MATCH_TEMPLATE=JACC.array(zeros(UInt8,5188))
const SCAN_TEMPLATE=JACC.array(zeros(UInt8,2048))
function run_match!(input,fs,ds,tf,td,nb);JACC.parallel_for(JACC.launch_spec(threads=128,blocks=nb,shmem_size=5188,sync=false),nb*128,match_kernel,input,fs,ds,tf,td,MATCH_TEMPLATE);end
function run_scan!(input,out,sums,n,groups);JACC.parallel_for(JACC.launch_spec(threads=128,blocks=groups,shmem_size=2048,sync=false),groups*128,scan_kernel,input,out,sums,n,SCAN_TEMPLATE);end
function run_add!(out,offsets,n);JACC.parallel_for(JACC.launch_spec(threads=128,blocks=cld(n,128),shmem_size=0,sync=false),cld(n,128)*128,add_kernel,out,offsets,n);end
function run_pack!(fo,doo,tf,td,pf,pd,nb);JACC.parallel_for(JACC.launch_spec(threads=128,blocks=nb,shmem_size=0,sync=false),nb*128,pack_kernel,fo,doo,tf,td,pf,pd);end
function run_decode!(out,fo,doo,pf,pd,nb);JACC.parallel_for(JACC.launch_spec(threads=32,blocks=cld(nb,32),shmem_size=0,sync=false),cld(nb,32)*32,decode_kernel,out,fo,doo,pf,pd,nb);end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
'''
(p/'lzss-julia-JACC'/'main.jl').write_text(s)
