from pathlib import Path
import json
from translation_templates import PCR,CR,SWEEP,TRANS
B=Path(__file__).resolve().parent;R=B.parents[1];C=B/'tridiagonal-julia-KA';C.mkdir(exist_ok=True)
s='''using CUDA,KernelAbstractions,StaticArrays
const BACKEND=CUDA.CUDABackend()
# NVIDIA/UC Davis source notice in ../NOTICE.source.
@inline load4(a,i)=SVector{4,Float32}(@inbounds(a[i+1]),@inbounds(a[i+2]),@inbounds(a[i+3]),@inbounds(a[i+4]))
@inline function store4!(a,i,v)
 @inbounds for k in 1:4;a[i+k]=v[k];end
 nothing
end
'''
for name,body in [('pcr',PCR),('cr',CR)]:
 s+=f'@kernel function {name}!(ad,bd,cd,dd,xd,::Val{{N}},m,::Val{{OPT}}) where {{N,OPT}}\n n=Int32(N);iterations=Int32(trailing_zeros(UInt32(N÷2)));lid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);sh=@localmem Float32 (5*(N+1),)\n'+body.replace('BARRIER','@synchronize')+'\nend\n'
s+='@kernel function sweep!(ad,bd,cd,dd,xd,wd,::Val{N},m,::Val{MODE}) where {N,MODE}\n n=Int32(N);globalid=Int32(@index(Global,Linear))-Int32(1)\n'+SWEEP+'\nend\n'
s+='@kernel function trans!(dst,src,width,height)\n bx=Int32(@index(Group,NTuple)[1])-Int32(1);by=Int32(@index(Group,NTuple)[2])-Int32(1);tx=Int32(@index(Local,NTuple)[1])-Int32(1);ty=Int32(@index(Local,NTuple)[2])-Int32(1);sh=@localmem Float32 (272,)\n'+TRANS.replace('BARRIER','@synchronize')+'\nend\n'
s+='''function launch(mode,a,b,c,d,x,w,n,m)
 if mode<2
  pcr!(BACKEND,n)(a,b,c,d,x,Val(n),Int32(m),Val(mode);ndrange=n*m)
 elseif mode<4
  cr!(BACKEND,n÷2)(a,b,c,d,x,Val(n),Int32(m),Val(mode-2);ndrange=n÷2*m)
 else
  th=mode==8 ? 64 : 256;bl=cld(m,256)*(mode==8 ? 4 : 1)
  sweep!(BACKEND,th)(a,b,c,d,x,w,Val(n),Int32(m),Val(mode);ndrange=bl*th)
 end
 nothing
end
function transpose_launch(dst,src,width,height)
 trans!(BACKEND,(16,16))(dst,src,Int32(width),Int32(height);ndrange=(width,height))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(v->begin a=KernelAbstractions.allocate(BACKEND,Float32,length(v));copyto!(a,v);a end,launch,transpose_launch)
''';(C/'main.jl').write_text(s)

start=s.index('@kernel function sweep!');end=s.index('@kernel function trans!',start);body=s[start:end];body=body.replace('@kernel function sweep!','@inline function sweep_body!').replace('n=Int32(N);globalid=Int32(@index(Global,Linear))-Int32(1)','n=Int32(N)').replace('::Val{MODE}) where','::Val{MODE},globalid) where');body+='@kernel function sweep!(ad,bd,cd,dd,xd,wd,vn,m,vm)\n globalid=Int32(@index(Global,Linear))-Int32(1)\n sweep_body!(ad,bd,cd,dd,xd,wd,vn,m,vm,globalid)\nend\n';s=s[:start]+body+s[end:];s=s.replace('(i0>=width||j0>=height)&&return nothing','# Host precondition: complete16x16tiles; all launched threads valid.').replace('(i1>=height||j1>=width)&&return nothing','# Source early-return conditions are uniformly false in this named regime.');s=s.replace('function transpose_launch(dst,src,width,height)\n','function transpose_launch(dst,src,width,height)\n @assert width%16==0 && height%16==0\n');(C/'main.jl').write_text(s)
