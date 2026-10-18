include("../study_common.jl")
@inline gid()=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
function extract_cuda(N,I);extract_site!(gid(),N,I);return;end
function prepare_cuda(N,I,s,s2);prepare_site!(gid(),N,I,s,s2);return;end
function compress_cuda(N,I);compress_site!(gid(),N,I);return;end
function coefficient_cuda(lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I);srad_site!(gid(),lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I);return;end
function update_cuda(lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I);update_site!(gid(),lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I);return;end
function reduce_cuda(N,no,mul,s,s2)
 tx=Int32(threadIdx().x-1);bx=Int32(blockIdx().x-1);nb=Int32(gridDim().x);ei=bx*Int32(256)+tx;nf=Int32(256)-(nb*Int32(256)-no)
 shared=CUDA.@cuStaticSharedMem(Float32,512)
 if ei<no;shared[tx+1]=s[ei*mul+1];shared[256+tx+1]=s2[ei*mul+1];end
 sync_threads()
 full=nf==256||bx!=nb-1;df=Int32(0)
 if full;df=256;else;i=Int32(2);while i<=256;if nf>=i;df=i;end;i*=2;end;end
 span=Int32(2)
 while span<=df
  if rem(tx+1,span)==0&&tx<df;shared[tx+1]+=shared[tx-div(span,2)+1];shared[256+tx+1]+=shared[256+tx-div(span,2)+1];end
  sync_threads();span*=2
 end
 if tx==df-1
  i=bx*Int32(256)+df;limit=bx*Int32(256)+(full ? Int32(256) : nf)
  while i<limit;shared[tx+1]+=s[i*mul+1];shared[256+tx+1]+=s2[i*mul+1];i+=1;end
  s[bx*mul*Int32(256)+1]=shared[tx+1];s2[bx*mul*Int32(256)+1]=shared[256+tx+1]
 end
 return
end
ex(c)=@cuda threads=256 blocks=c.blocks extract_cuda(c.N,c.I.data)
prep(c)=@cuda threads=256 blocks=c.blocks prepare_cuda(c.N,c.I.data,c.s.data,c.s2.data)
red(c,no,mul,b)=@cuda threads=256 blocks=b reduce_cuda(c.N,no,mul,c.s.data,c.s2.data)
coef(c,q0)=@cuda threads=256 blocks=c.blocks coefficient_cuda(c.lambda,c.nr,c.nc,c.N,c.iN.data,c.iS.data,c.jE.data,c.jW.data,c.dN.data,c.dS.data,c.dE.data,c.dW.data,q0,c.c.data,c.I.data)
upd(c)=@cuda threads=256 blocks=c.blocks update_cuda(c.lambda,c.nr,c.nc,c.N,c.iN.data,c.iS.data,c.jE.data,c.jW.data,c.dN.data,c.dS.data,c.dE.data,c.dW.data,c.c.data,c.I.data)
comp(c)=@cuda threads=256 blocks=c.blocks compress_cuda(c.N,c.I.data)
srad_study((extract=ex,prepare=prep,reduce=red,coefficient=coef,update=upd,compress=comp))
