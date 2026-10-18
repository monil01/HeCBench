from pathlib import Path
import json
from translation_templates import PCR,CR,SWEEP,TRANS
B=Path(__file__).resolve().parent;R=B.parents[1];C=B/'tridiagonal-julia-cuda.jl';C.mkdir(exist_ok=True)
helper='''using CUDA,StaticArrays
# Source notice in ../NOTICE.source; algorithms follow original NVIDIA/UC Davis kernels.
@inline load4(a,i)=SVector{4,Float32}(@inbounds(a[i+1]),@inbounds(a[i+2]),@inbounds(a[i+3]),@inbounds(a[i+4]))
@inline function store4!(a,i,v)
 @inbounds for k in 1:4;a[i+k]=v[k];end
 nothing
end
'''
s=helper
for name,body in [('pcr',PCR),('cr',CR)]:
 s+=f'function {name}!(ad,bd,cd,dd,xd,::Val{{N}},m,::Val{{OPT}}) where {{N,OPT}}\n n=Int32(N);iterations=Int32(trailing_zeros(UInt32(N÷2)));lid=Int32(CUDA.threadIdx().x)-Int32(1);bid=Int32(CUDA.blockIdx().x)-Int32(1);sh=CUDA.CuDynamicSharedArray(Float32,5*(N+1))\n'+body.replace('BARRIER','CUDA.sync_threads()')+'\n return nothing\nend\n'
s+='function sweep!(ad,bd,cd,dd,xd,wd,::Val{N},m,::Val{MODE}) where {N,MODE}\n n=Int32(N);globalid=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-Int32(1)\n'+SWEEP+'\n return nothing\nend\n'
s+='function trans!(dst,src,width,height)\n bx=Int32(CUDA.blockIdx().x)-Int32(1);by=Int32(CUDA.blockIdx().y)-Int32(1);tx=Int32(CUDA.threadIdx().x)-Int32(1);ty=Int32(CUDA.threadIdx().y)-Int32(1);sh=CUDA.CuStaticSharedArray(Float32,272)\n'+TRANS.replace('BARRIER','CUDA.sync_threads()')+'\n return nothing\nend\n'
s+='''function launch(mode,a,b,c,d,x,w,n,m)
 if mode<2
  CUDA.@cuda threads=n blocks=m shmem=5*(n+1)*4 pcr!(a,b,c,d,x,Val(n),Int32(m),Val(mode))
 elseif mode<4
  CUDA.@cuda threads=n÷2 blocks=m shmem=5*(n+1)*4 cr!(a,b,c,d,x,Val(n),Int32(m),Val(mode-2))
 else
  th=mode==8 ? 64 : 256;bl=cld(m,256)*(mode==8 ? 4 : 1)
  CUDA.@cuda threads=th blocks=bl sweep!(a,b,c,d,x,w,Val(n),Int32(m),Val(mode))
 end
 nothing
end
function transpose_launch(dst,src,width,height)
 CUDA.@cuda threads=(16,16) blocks=(width÷16,height÷16) trans!(dst,src,Int32(width),Int32(height))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch,transpose_launch)
'''
(C/'main.jl').write_text(s);(B/'main.jl').write_text('include(joinpath(@__DIR__,"tridiagonal-julia-cuda.jl","main.jl"))\n')
