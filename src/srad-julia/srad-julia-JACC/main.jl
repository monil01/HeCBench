using JACC
include("../study_common.jl")
const TEMPLATE=CUDA.zeros(Float32,512)
extract_jacc(i,N,I)=extract_site!(Int32(i)-Int32(1),N,I)
prepare_jacc(i,N,I,s,s2)=prepare_site!(Int32(i)-Int32(1),N,I,s,s2)
compress_jacc(i,N,I)=compress_site!(Int32(i)-Int32(1),N,I)
coefficient_jacc(i,lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I)=srad_site!(Int32(i)-Int32(1),lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,q0,c,I)
update_jacc(i,lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I)=update_site!(Int32(i)-Int32(1),lambda,nr,nc,N,iN,iS,jE,jW,dN,dS,dE,dW,c,I)
function reduce_jacc(i,N,no,mul,s,s2,nb::Int32,template)
 ei=Int32(i)-Int32(1);tx=rem(ei,Int32(256));bx=div(ei,Int32(256));nf=Int32(256)-(nb*Int32(256)-no)
 shared=JACC.shared(template)
 if ei<no;shared[tx+1]=s[ei*mul+1];shared[256+tx+1]=s2[ei*mul+1];end
 JACC.sync_workgroup()
 full=nf==256||bx!=nb-1;df=Int32(0)
 if full;df=256;else;j=Int32(2);while j<=256;if nf>=j;df=j;end;j*=2;end;end
 span=Int32(2)
 while span<=df
  if rem(tx+1,span)==0&&tx<df;shared[tx+1]+=shared[tx-div(span,2)+1];shared[256+tx+1]+=shared[256+tx-div(span,2)+1];end
  JACC.sync_workgroup();span*=2
 end
 if tx==df-1
  j=bx*Int32(256)+df;limit=bx*Int32(256)+(full ? Int32(256) : nf)
  while j<limit;shared[tx+1]+=s[j*mul+1];shared[256+tx+1]+=s2[j*mul+1];j+=1;end
  s[bx*mul*Int32(256)+1]=shared[tx+1];s2[bx*mul*Int32(256)+1]=shared[256+tx+1]
 end
 nothing
end
const SPECS=Dict((b,sh)=>JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=b,shmem_size=sh,sync=false) for b in [1,2,3,4,259],sh in [0,2048])
ex(c)=JACC.parallel_for(extract_jacc,SPECS[(c.blocks,0)],c.blocks*256,c.N,c.I.data)
prep(c)=JACC.parallel_for(prepare_jacc,SPECS[(c.blocks,0)],c.blocks*256,c.N,c.I.data,c.s.data,c.s2.data)
red(c,no,mul,b)=JACC.parallel_for(reduce_jacc,SPECS[(b,2048)],b*256,c.N,no,mul,c.s.data,c.s2.data,Int32(b),TEMPLATE)
coef(c,q0)=JACC.parallel_for(coefficient_jacc,SPECS[(c.blocks,0)],c.blocks*256,c.lambda,c.nr,c.nc,c.N,c.iN.data,c.iS.data,c.jE.data,c.jW.data,c.dN.data,c.dS.data,c.dE.data,c.dW.data,q0,c.c.data,c.I.data)
upd(c)=JACC.parallel_for(update_jacc,SPECS[(c.blocks,0)],c.blocks*256,c.lambda,c.nr,c.nc,c.N,c.iN.data,c.iS.data,c.jE.data,c.jW.data,c.dN.data,c.dS.data,c.dE.data,c.dW.data,c.c.data,c.I.data)
comp(c)=JACC.parallel_for(compress_jacc,SPECS[(c.blocks,0)],c.blocks*256,c.N,c.I.data)
srad_study((extract=ex,prepare=prep,reduce=red,coefficient=coef,update=upd,compress=comp))
