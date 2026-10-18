from pathlib import Path
import json
from translation_templates import PCR,CR,SWEEP,TRANS
B=Path(__file__).resolve().parent;R=B.parents[1];C=B/'tridiagonal-julia-JACC';C.mkdir(exist_ok=True)
s='''using CUDA,StaticArrays
import JACC
JACC.@init_backend
const TEMPLATES=Dict(n=>JACC.zeros(Float32,5*(n+1)) for n in [16,32,64,128])
const TRANS_TEMPLATE=JACC.zeros(Float32,272)
# NVIDIA/UC Davis source notice in ../NOTICE.source.
@inline load4(a,i)=SVector{4,Float32}(@inbounds(a[i+1]),@inbounds(a[i+2]),@inbounds(a[i+3]),@inbounds(a[i+4]))
@inline function store4!(a,i,v)
 @inbounds for k in 1:4;a[i+k]=v[k];end
 nothing
end
'''
for name,body in [('pcr',PCR),('cr',CR)]:
 th='N'if name=='pcr'else'N÷2'
 s+=f'function {name}!(index,ad,bd,cd,dd,xd,::Val{{N}},m,::Val{{OPT}},template) where {{N,OPT}}\n n=Int32(N);iterations=Int32(trailing_zeros(UInt32(N÷2)));lid=Int32((index-1)%({th}));bid=Int32((index-1)÷({th}));sh=JACC.shared(template)\n'+body.replace('BARRIER','JACC.sync_workgroup()')+'\n return nothing\nend\n'
s+='function sweep!(index,ad,bd,cd,dd,xd,wd,::Val{N},m,::Val{MODE}) where {N,MODE}\n n=Int32(N);globalid=Int32(index)-Int32(1)\n'+SWEEP+'\n return nothing\nend\n'
s+='function trans!(i,j,dst,src,width,height,template)\n bx=Int32((i-1)÷16);by=Int32((j-1)÷16);tx=Int32((i-1)%16);ty=Int32((j-1)%16);sh=JACC.shared(template)\n'+TRANS.replace('BARRIER','JACC.sync_workgroup()')+'\n return nothing\nend\n'
s+='''function launch(mode,a,b,c,d,x,w,n,m)
 if mode<4
  th=mode<2 ? n : n÷2;spec=JACC.launch_spec(;threads=th,blocks=m,shmem_size=5*(n+1)*4,sync=false)
  if mode<2;JACC.parallel_for(spec,m*th,pcr!,a,b,c,d,x,Val(n),Int32(m),Val(mode),TEMPLATES[n])
  else;JACC.parallel_for(spec,m*th,cr!,a,b,c,d,x,Val(n),Int32(m),Val(mode-2),TEMPLATES[n]);end
 else
  th=mode==8 ? 64 : 256;bl=cld(m,256)*(mode==8 ? 4 : 1);spec=JACC.launch_spec(;threads=th,blocks=bl,shmem_size=0,sync=false)
  JACC.parallel_for(spec,bl*th,sweep!,a,b,c,d,x,w,Val(n),Int32(m),Val(mode))
 end
 nothing
end
function transpose_launch(dst,src,width,height)
 spec=JACC.launch_spec(;threads=(16,16),blocks=(width÷16,height÷16),shmem_size=272*4,sync=false)
 JACC.parallel_for(spec,(width,height),trans!,dst,src,Int32(width),Int32(height),TRANS_TEMPLATE)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch,transpose_launch)
''';(C/'main.jl').write_text(s)
