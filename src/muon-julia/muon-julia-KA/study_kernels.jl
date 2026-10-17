using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
@kernel function momentum_ka!(mom,g,x,beta,n);momentum_body!((Int32(@index(Global,Linear))-Int32(1)),mom,g,x,beta,n);end
@kernel function normalize_ka!(x,ss,n);inv=1f0/(sqrt(@inbounds ss[1])+1f-7);vector_body!((Int32(@index(Global,Linear))-Int32(1)),x,x,x,inv,0f0,n,Int32(0));end
@kernel function combine_ka!(b,a,a2,n);vector_body!((Int32(@index(Global,Linear))-Int32(1)),b,a,a2,NS_B,NS_C,n,Int32(1));end
@kernel function update_ka!(x,bx,n);vector_body!((Int32(@index(Global,Linear))-Int32(1)),x,x,bx,NS_A,0f0,n,Int32(2));end
@kernel function parameter_ka!(p,x,lr,wd,scale,n);parameter_body!((Int32(@index(Global,Linear))-Int32(1)),p,x,lr,wd,scale,n);end
@kernel function sumsq_ka!(x,out,n)
 shared=@localmem Float32 (256,);tid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);v=0f0;n4=n>>2
 for j in bid*Int32(256)+tid:Int32(65536):n4-Int32(1)
  a=decode_bf(@inbounds x[j*4+1]);b=decode_bf(@inbounds x[j*4+2]);c=decode_bf(@inbounds x[j*4+3]);d=decode_bf(@inbounds x[j*4+4]);v+=a*a+b*b+c*c+d*d
 end
 if bid==0&&tid==0
  for i in n4*Int32(4)+Int32(1):n;t=decode_bf(@inbounds x[i]);v+=t*t;end
 end
 @inbounds shared[tid+1]=v;@synchronize()
 for stride in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride;@inbounds shared[tid+1]+=shared[tid+stride+1];end
  @synchronize()
 end
 if tid==0;CUDA.atomic_add!(pointer(out,1),@inbounds(shared[1]));end
end
@kernel function momentum_transpose_ka!(mom,g,x,beta,P,Q)
 tile=@localmem UInt16 (1024,);localidx=@index(Local,NTuple);groupidx=@index(Group,NTuple);tx=Int32(localidx[1])-Int32(1);ty=Int32(localidx[2])-Int32(1);bx=Int32(groupidx[1])-Int32(1);by=Int32(groupidx[2])-Int32(1);col=bx*32+tx;row=by*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<Q&&row+j<P;@inbounds tile[(ty+j)*32+((col-bx*32)⊻(ty+j))+1]=momentum_value!(mom,g,(row+j)*Q+col+1,beta);end
 end
 @synchronize();co=by*32+tx;ro=bx*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if co<P&&ro+j<Q;@inbounds x[(ro+j)*P+co+1]=tile[tx*32+(tx⊻(ty+j))+1];end
 end
end
@kernel function transpose_parameter_ka!(p,x,lr,wd,scale,P,Q)
 tile=@localmem UInt16 (1024,);localidx=@index(Local,NTuple);groupidx=@index(Group,NTuple);tx=Int32(localidx[1])-Int32(1);ty=Int32(localidx[2])-Int32(1);bx=Int32(groupidx[1])-Int32(1);by=Int32(groupidx[2])-Int32(1);col=bx*32+tx;row=by*32+ty
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if col<P&&row+j<Q;@inbounds tile[(ty+j)*32+(tx⊻(ty+j))+1]=x[(row+j)*P+col+1];end
 end
 @synchronize();co=by*32+tx;ro=bx*32+ty;decay=1f0-lr*wd
 for j in (Int32(0),Int32(8),Int32(16),Int32(24))
  if co<Q&&ro+j<P;i=(ro+j)*Q+co+1;@inbounds p[i]=muladd(-lr,scale*decode_bf(tile[tx*32+(tx⊻(ty+j))+1]),p[i]*decay);end
 end
end
