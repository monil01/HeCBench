using KernelAbstractions
include("../study_common.jl")
@kernel function extract_ka(N,I)
 i=Int32(@index(Global,Linear))-Int32(1);extract_site!(i,N,I)
end
@kernel function prepare_ka(N,I,s,s2)
 i=Int32(@index(Global,Linear))-Int32(1);prepare_site!(i,N,I,s,s2)
end
@kernel function compress_ka(N,I)
 i=Int32(@index(Global,Linear))-Int32(1);compress_site!(i,N,I)
end
@kernel function coefficient_ka(lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I)
 i=Int32(@index(Global,Linear))-Int32(1);srad_site!(i,lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I)
end
@kernel function update_ka(lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I)
 i=Int32(@index(Global,Linear))-Int32(1);update_site!(i,lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I)
end
@kernel function reduce_ka(N,no,mul,s,s2,nb::Int32)
 tx=Int32(@index(Local,Linear))-Int32(1);bx=Int32(@index(Group,Linear))-Int32(1);ei=bx*Int32(256)+tx;nf=Int32(256)-(nb*Int32(256)-no)
 shared=@localmem Float32 (512,)
 if ei<no;shared[tx+1]=s[ei*mul+1];shared[256+tx+1]=s2[ei*mul+1];end
 @synchronize
 full=nf==256||bx!=nb-1;df=Int32(0)
 if full;df=256;else;i=Int32(2);while i<=256;if nf>=i;df=i;end;i*=2;end;end
 span=Int32(2)
 while span<=df
  if rem(tx+1,span)==0&&tx<df;shared[tx+1]+=shared[tx-div(span,2)+1];shared[256+tx+1]+=shared[256+tx-div(span,2)+1];end
  @synchronize
  span*=2
 end
 if tx==df-1
  i=bx*Int32(256)+df;limit=bx*Int32(256)+(full ? Int32(256) : nf)
  while i<limit;shared[tx+1]+=s[i*mul+1];shared[256+tx+1]+=s2[i*mul+1];i+=1;end
  s[bx*mul*Int32(256)+1]=shared[tx+1];s2[bx*mul*Int32(256)+1]=shared[256+tx+1]
 end
end
const KEX=extract_ka(CUDABackend(),256)
const KPREP=prepare_ka(CUDABackend(),256)
const KCOMP=compress_ka(CUDABackend(),256)
const KCOEF=coefficient_ka(CUDABackend(),256)
const KUPDATE=update_ka(CUDABackend(),256)
const KRED=reduce_ka(CUDABackend(),256)
ex(c)=KEX(c.N,c.I.data;ndrange=c.blocks*256)
prep(c)=KPREP(c.N,c.I.data,c.s.data,c.s2.data;ndrange=c.blocks*256)
red(c,no,mul,b)=KRED(c.N,no,mul,c.s.data,c.s2.data,Int32(b);ndrange=b*256)
coef(c,q0)=KCOEF(c.lambda,c.nr,c.nc,c.N,c.iN.data,c.iS.data,c.jE.data,c.jW.data,c.dN.data,c.dS.data,c.dE.data,c.dW.data,q0,c.c.data,c.I.data;ndrange=c.blocks*256)
upd(c)=KUPDATE(c.lambda,c.nr,c.nc,c.N,c.iN.data,c.iS.data,c.jE.data,c.jW.data,c.dN.data,c.dS.data,c.dE.data,c.dW.data,c.c.data,c.I.data;ndrange=c.blocks*256)
comp(c)=KCOMP(c.N,c.I.data;ndrange=c.blocks*256)
srad_study((extract=ex,prepare=prep,reduce=red,coefficient=coef,update=upd,compress=comp))
