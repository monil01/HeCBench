include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const MOM=momentum_ka!(BACKEND,256);const NORM=normalize_ka!(BACKEND,256);const COMB=combine_ka!(BACKEND,256);const UP=update_ka!(BACKEND,256);const PARAM=parameter_ka!(BACKEND,256);const SUM=sumsq_ka!(BACKEND,256);const MT=momentum_transpose_ka!(BACKEND,(32,8));const TP=transpose_parameter_ka!(BACKEND,(32,8))
function run(p,m,g,x,a,a2,b,bx,ss,P,Q,scale,gemm)
 n=P*Q;mm=min(P,Q);nn=max(P,Q);gn=cld(cld(n,4),256)*256;gm=cld(cld(mm*mm,4),256)*256
 if P>Q;MT(m,g,x,.95f0,P,Q;ndrange=(cld(Q,32)*32,cld(P,32)*8))
 else;MOM(m,g,x,.95f0,n;ndrange=gn);end
 CUDA.fill!(ss,0f0);SUM(x,ss,n;ndrange=65536);NORM(x,ss,n;ndrange=gn)
 for _ in 1:5
  gemm(0,1,mm,mm,nn,x,x,a);gemm(0,0,mm,mm,mm,a,a,a2);COMB(b,a,a2,mm*mm;ndrange=gm);gemm(0,0,mm,nn,mm,b,x,bx);UP(x,bx,n;ndrange=gn)
 end
 if P>Q;TP(p,x,.02f0,0f0,scale,P,Q;ndrange=(cld(P,32)*32,cld(Q,32)*8))
 else;PARAM(p,x,.02f0,0f0,scale,n;ndrange=gn);end
 return
end
study(run)
