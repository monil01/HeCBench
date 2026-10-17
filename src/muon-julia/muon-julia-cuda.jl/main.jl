include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run(p,m,g,x,a,a2,b,bx,ss,P,Q,scale,gemm)
 n=P*Q;mm=min(P,Q);nn=max(P,Q);gn=cld(cld(n,4),256);gm=cld(cld(mm*mm,4),256)
 if P>Q;@cuda threads=(32,8) blocks=(cld(Q,32),cld(P,32)) momentum_transpose_cuda!(m,g,x,.95f0,P,Q)
 else;@cuda threads=256 blocks=gn momentum_cuda!(m,g,x,.95f0,n);end
 CUDA.fill!(ss,0f0);@cuda threads=256 blocks=256 sumsq_cuda!(x,ss,n);@cuda threads=256 blocks=gn normalize_cuda!(x,ss,n)
 for _ in 1:5
  gemm(0,1,mm,mm,nn,x,x,a);gemm(0,0,mm,mm,mm,a,a,a2);@cuda threads=256 blocks=gm combine_cuda!(b,a,a2,mm*mm);gemm(0,0,mm,nn,mm,b,x,bx);@cuda threads=256 blocks=gn update_cuda!(x,bx,n)
 end
 if P>Q;@cuda threads=(32,8) blocks=(cld(P,32),cld(Q,32)) transpose_parameter_cuda!(p,x,.02f0,0f0,scale,P,Q)
 else;@cuda threads=256 blocks=gn parameter_cuda!(p,x,.02f0,0f0,scale,n);end
 return
end
study(run)
