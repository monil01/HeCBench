using CUDA
include(joinpath(@__DIR__,"study_kernel_body.jl"))
@inline linear0()=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
function momentum_cuda!(mom,g,x,beta,n);momentum_body!(linear0(),mom,g,x,beta,n);return;end
function normalize_cuda!(x,ss,n);inv=1f0/(sqrt(@inbounds ss[1])+1f-7);vector_body!(linear0(),x,x,x,inv,0f0,n,Int32(0));return;end
function combine_cuda!(b,a,a2,n);vector_body!(linear0(),b,a,a2,NS_B,NS_C,n,Int32(1));return;end
function update_cuda!(x,bx,n);vector_body!(linear0(),x,x,bx,NS_A,0f0,n,Int32(2));return;end
function parameter_cuda!(p,x,lr,wd,scale,n);parameter_body!(linear0(),p,x,lr,wd,scale,n);return;end
function sumsq_cuda!(x,out,n)
 shared=CuStaticSharedArray(Float32,256);tid=Int32(threadIdx().x)-Int32(1);bid=Int32(blockIdx().x)-Int32(1);v=0f0;n4=n>>2
 for j in bid*Int32(256)+tid:Int32(65536):n4-Int32(1)
  a=decode_bf(@inbounds x[j*4+1]);b=decode_bf(@inbounds x[j*4+2]);c=decode_bf(@inbounds x[j*4+3]);d=decode_bf(@inbounds x[j*4+4]);v+=a*a+b*b+c*c+d*d
 end
 if bid==0&&tid==0
  for i in n4*Int32(4)+Int32(1):n;t=decode_bf(@inbounds x[i]);v+=t*t;end
 end
 @inbounds shared[tid+1]=v;sync_threads()
 for stride in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride;@inbounds shared[tid+1]+=shared[tid+stride+1];end
  sync_threads()
 end
 if tid==0;CUDA.atomic_add!(pointer(out,1),@inbounds(shared[1]));end
 return
end
function momentum_transpose_cuda!(mom,g,x,beta,P,Q)
 tile=CuStaticSharedArray(UInt16,1024);tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);bx=Int32(blockIdx().x)-Int32(1);by=Int32(blockIdx().y)-Int32(1);col=bx*32+tx;row=by*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<Q&&row+j<P;@inbounds tile[(ty+j)*32+((col-bx*32)⊻(ty+j))+1]=momentum_value!(mom,g,(row+j)*Q+col+1,beta);end
 end
 sync_threads();co=by*32+tx;ro=bx*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if co<P&&ro+j<Q;@inbounds x[(ro+j)*P+co+1]=tile[tx*32+(tx⊻(ty+j))+1];end
 end
 return
end
function transpose_parameter_cuda!(p,x,lr,wd,scale,P,Q)
 tile=CuStaticSharedArray(UInt16,1024);tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);bx=Int32(blockIdx().x)-Int32(1);by=Int32(blockIdx().y)-Int32(1);col=bx*32+tx;row=by*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<P&&row+j<Q;@inbounds tile[(ty+j)*32+(tx⊻(ty+j))+1]=x[(row+j)*P+col+1];end
 end
 sync_threads();co=by*32+tx;ro=bx*32+ty;decay=1f0-lr*wd
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if co<Q&&ro+j<P;i=(ro+j)*Q+co+1;@inbounds p[i]=muladd(-lr,scale*decode_bf(tile[tx*32+(tx⊻(ty+j))+1]),p[i]*decay);end
 end
 return
end
