using Libdl
const LIB_MUON="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
const SHAPES=((16,32,false),(32,16,false),(17,33,true),(33,17,true))
function fixture(P,Q,dense)
 p=Float32[Float32(mod(i*11,37)-18)/64f0 for i in 0:P*Q-1];m=Float32[dense ? Float32(mod(i*13,29)-14)/64f0 : (i÷Q==i%Q ? Float32(i%Q+1)/64f0 : 0f0) for i in 0:P*Q-1];g=Float32[dense ? Float32(mod(i*7,31)-15)/128f0 : (i÷Q==i%Q ? Float32(i%Q+1)/128f0 : 0f0) for i in 0:P*Q-1];return p,m,g
end
function rowgemm(fn,h,opA,opB,M,N,K,A,B,C)
 alpha=Ref{Float32}(1);beta=Ref{Float32}(0);lda=opA==0 ? K : M;ldb=opB==0 ? N : K
 GC.@preserve alpha beta A B C begin
  status=ccall(fn,Cint,(Ptr{Cvoid},Cint,Cint,Cint,Cint,Cint,Ptr{Cvoid},CuPtr{Cvoid},Cint,Cint,CuPtr{Cvoid},Cint,Cint,Ptr{Cvoid},CuPtr{Cvoid},Cint,Cint,Cint,Cint),h,opB,opA,N,M,K,alpha,pointer(B),14,ldb,pointer(A),14,lda,beta,pointer(C),14,N,68,99)
 end
 status==0||Base.error("BF16GemmStatus",status);return
end
function study(run)
 lib=Libdl.dlopen(LIB_MUON);href=Ref{Ptr{Cvoid}}(C_NULL);ccall(Libdl.dlsym(lib,:cublasCreate_v2),Cint,(Ref{Ptr{Cvoid}},),href)==0||error("CreateFailed")
 # Bind raw vendor calls to the same stream as authored kernels; necessary inter-stage dependency, not just timer completion.
 ccall(Libdl.dlsym(lib,:cublasSetStream_v2),Cint,(Ptr{Cvoid},Ptr{Cvoid}),href[],Ptr{Cvoid}(CUDA.stream().handle))==0||error("SetStreamFailed")
 fn=Libdl.dlsym(lib,:cublasGemmEx);cases=[];prefix=Float64[]
 for(P,Q,dense)in SHAPES
  p,m,g=fixture(P,Q,dense);scale=sqrt(max(1f0,Float32(P)/Float32(Q)));append!(prefix,Float64.(p));append!(prefix,Float64.(m));append!(prefix,Float64.(g));append!(prefix,Float64[P,Q,2,.95f0,.02f0,0f0,scale,dense]);mm=min(P,Q);n=P*Q
  push!(cases,(P,Q,p,m,CuArray(p),CuArray(m),CuArray(g),CUDA.zeros(UInt16,n),CUDA.zeros(UInt16,mm*mm),CUDA.zeros(UInt16,mm*mm),CUDA.zeros(UInt16,mm*mm),CUDA.zeros(UInt16,n),CUDA.zeros(Float32,1),scale))
 end
 function case!(c)
  P,Q,p0,m0,p,m,g,x,a,a2,b,bx,ss,scale=c;copyto!(p,p0);copyto!(m,m0);CUDA.synchronize(CUDA.context());t=time_ns()
  for _ in 1:2;run(p,m,g,x,a,a2,b,bx,ss,Int32(P),Int32(Q),scale,(oa,ob,M,N,K,A,B,C)->rowgemm(fn,href[],oa,ob,M,N,K,A,B,C));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9/2;out=Float64[]
  append!(out,Float64.(Array(p)));append!(out,Float64.(Array(m)))
  for data in (x,a,a2,b,bx);append!(out,Float64.(decode_bf.(Array(data))));end
  append!(out,Float64.(Array(ss)));return out,elapsed
 end
 for c in cases;case!(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.
  for c in cases;got,t=case!(c);append!(out,got);elapsed+=t;end
  ok=all(isfinite,out);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"timing_region\":\"sum_four_two_step_optimizer_means\",\"jit_excluded\":true,\"internal_repeats\":2,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("NonfiniteMuonState")
 end
 ccall(Libdl.dlsym(lib,:cublasDestroy_v2),Cint,(Ptr{Cvoid},),href[])
end
