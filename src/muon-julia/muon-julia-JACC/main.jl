include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const SUM_TEMPLATE=CUDA.zeros(Float32,256)
const TILE_TEMPLATE=CUDA.zeros(UInt16,1024)
function launch(f,N,args...;threads=256,blocks=cld(N,256),shared=0)
 JACC.parallel_for(JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=shared,sync=false),N,f,args...);return
end
function run(p,m,g,x,a,a2,b,bx,ss,P,Q,scale,gemm)
 n=P*Q;mm=min(P,Q);nn=max(P,Q);gn=cld(cld(n,4),256)*256;gm=cld(cld(mm*mm,4),256)*256
 if P>Q;blocks=(cld(Q,32),cld(P,32));launch(momentum_transpose_jacc!,(blocks[1]*32,blocks[2]*8),m,g,x,.95f0,P,Q,TILE_TEMPLATE;threads=(32,8),blocks=blocks,shared=2048)
 else;launch(momentum_jacc!,gn,m,g,x,.95f0,n);end
 CUDA.fill!(ss,0f0);launch(sumsq_jacc!,65536,x,ss,n,SUM_TEMPLATE;threads=256,blocks=256,shared=1024);launch(normalize_jacc!,gn,x,ss,n)
 for _ in 1:5
  gemm(0,1,mm,mm,nn,x,x,a);gemm(0,0,mm,mm,mm,a,a,a2);launch(combine_jacc!,gm,b,a,a2,mm*mm);gemm(0,0,mm,nn,mm,b,x,bx);launch(update_jacc!,gn,x,bx,n)
 end
 if P>Q;blocks=(cld(P,32),cld(Q,32));launch(transpose_parameter_jacc!,(blocks[1]*32,blocks[2]*8),p,x,.02f0,0f0,scale,P,Q,TILE_TEMPLATE;threads=(32,8),blocks=blocks,shared=2048)
 else;launch(parameter_jacc!,gn,p,x,.02f0,0f0,scale,n);end
 return
end
study(run)
