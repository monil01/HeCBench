using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
function momentum_jacc!(i,mom,g,x,beta,n);momentum_body!((Int32(i)-Int32(1)),mom,g,x,beta,n);return;end
function normalize_jacc!(i,x,ss,n);inv=1f0/(sqrt(@inbounds ss[1])+1f-7);vector_body!((Int32(i)-Int32(1)),x,x,x,inv,0f0,n,Int32(0));return;end
function combine_jacc!(i,b,a,a2,n);vector_body!((Int32(i)-Int32(1)),b,a,a2,NS_B,NS_C,n,Int32(1));return;end
function update_jacc!(i,x,bx,n);vector_body!((Int32(i)-Int32(1)),x,x,bx,NS_A,0f0,n,Int32(2));return;end
function parameter_jacc!(i,p,x,lr,wd,scale,n);parameter_body!((Int32(i)-Int32(1)),p,x,lr,wd,scale,n);return;end
function sumsq_jacc!(i,x,out,n,template)
 shared=JACC.shared(template);tid=(Int32(i)-Int32(1))%Int32(256);bid=(Int32(i)-Int32(1))÷Int32(256);v=0f0;n4=n>>2
 for j in bid*Int32(256)+tid:Int32(65536):n4-Int32(1)
  a=decode_bf(@inbounds x[j*4+1]);b=decode_bf(@inbounds x[j*4+2]);c=decode_bf(@inbounds x[j*4+3]);d=decode_bf(@inbounds x[j*4+4]);v+=a*a+b*b+c*c+d*d
 end
 if bid==0&&tid==0
  for i in n4*Int32(4)+Int32(1):n;t=decode_bf(@inbounds x[i]);v+=t*t;end
 end
 @inbounds shared[tid+1]=v;JACC.sync_workgroup()
 for stride in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride;@inbounds shared[tid+1]+=shared[tid+stride+1];end
  JACC.sync_workgroup()
 end
 if tid==0;CUDA.atomic_add!(pointer(out,1),@inbounds(shared[1]));end
 return
end
function momentum_transpose_jacc!(ix,iy,mom,g,x,beta,P,Q,template)
 tile=JACC.shared(template);tx=(Int32(ix)-Int32(1))%Int32(32);ty=(Int32(iy)-Int32(1))%Int32(8);bx=(Int32(ix)-Int32(1))÷Int32(32);by=(Int32(iy)-Int32(1))÷Int32(8);col=bx*32+tx;row=by*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<Q&&row+j<P;@inbounds tile[(ty+j)*32+((col-bx*32)⊻(ty+j))+1]=momentum_value!(mom,g,(row+j)*Q+col+1,beta);end
 end
 JACC.sync_workgroup();co=by*32+tx;ro=bx*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if co<P&&ro+j<Q;@inbounds x[(ro+j)*P+co+1]=tile[tx*32+(tx⊻(ty+j))+1];end
 end
 return
end
function transpose_parameter_jacc!(ix,iy,p,x,lr,wd,scale,P,Q,template)
 tile=JACC.shared(template);tx=(Int32(ix)-Int32(1))%Int32(32);ty=(Int32(iy)-Int32(1))%Int32(8);bx=(Int32(ix)-Int32(1))÷Int32(32);by=(Int32(iy)-Int32(1))÷Int32(8);col=bx*32+tx;row=by*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<P&&row+j<Q;@inbounds tile[(ty+j)*32+(tx⊻(ty+j))+1]=x[(row+j)*P+col+1];end
 end
 JACC.sync_workgroup();co=by*32+tx;ro=bx*32+ty;decay=1f0-lr*wd
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if co<Q&&ro+j<P;i=(ro+j)*Q+co+1;@inbounds p[i]=muladd(-lr,scale*decode_bf(tile[tx*32+(tx⊻(ty+j))+1]),p[i]*decay);end
 end
 return
end
