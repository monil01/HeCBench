include("lt_api.jl")
function lt_study(pipeline)
 CUDA.device!(0);CUDA.context();println("CUBLASLT_PROVENANCE version=",ccall((:cublasLtGetVersion,LT_LIB),Csize_t,())," path=",realpath(Libdl.dlpath(Libdl.dlopen(LT_LIB))));println("CUDART_PROVENANCE path=",realpath(Libdl.dlpath(Libdl.dlopen(RT_LIB))))
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==70696&&length(gold)==108032;pos=2;cases=Any[]
 function managed(T,values,bf=false)
  lead=256÷sizeof(T);sentinel=T==Int8 ? T(73) : bf ? encode_bf(16384) : T(16384);v=bf ? encode_bf.(values) : T.(values);host=vcat(fill(sentinel,lead),v,fill(sentinel,lead));base=Ref{Ptr{Cvoid}}();rtck(ccall((:cudaMallocManaged,RT_LIB),Cint,(Ref{Ptr{Cvoid}},Csize_t,Cuint),base,sizeof(host),1));d=base[]+256
  GC.@preserve host rtck(ccall((:cudaMemcpy,RT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),base[],pointer(host),sizeof(host),1))
  (base=base[],d=d,host=host,bf=bf,bytes=sizeof(host))
 end
 for ci in 1:27
  code,m,n,k,mode=Int.(pre[pos:pos+4]);pos+=5;T,S,CT,atype,ctype,scale,compute=raw_dtype(code);A=managed(T,pre[pos:pos+m*k-1],code==7);pos+=m*k;B=managed(T,pre[pos:pos+k*n-1],code==7);pos+=k*n;C=managed(S,pre[pos:pos+m*n-1],code==7);pos+=m*n
  push!(cases,(code=code,m=m,n=n,k=k,mode=mode,A=A,B=B,C=C,alpha=CT[1],beta=CT[0],atype=atype,ctype=ctype,scale=scale,compute=compute))
 end
 @assert pos==length(pre)+1
 handle=Ref{Ptr{Cvoid}}();ltck(ccall((:cublasLtCreate,LT_LIB),Cint,(Ref{Ptr{Cvoid}},),handle));wb=Ref{Ptr{Cvoid}}();rtck(ccall((:cudaMalloc,RT_LIB),Cint,(Ref{Ptr{Cvoid}},Csize_t),wb,33554432+512));workspace=wb[]+256;rtck(ccall((:cudaMemset,RT_LIB),Cint,(Ptr{Cvoid},Cint,Csize_t),wb[],165,256));rtck(ccall((:cudaMemset,RT_LIB),Cint,(Ptr{Cvoid},Cint,Csize_t),wb[]+256+33554432,165,256))
 function prefetch(c)
  for a in (c.A,c.B,c.C);rtck(ccall((:cudaMemPrefetchAsync,RT_LIB),Cint,(Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),a.base,a.bytes,0,C_NULL));end
  rtck(ccall((:cudaDeviceSynchronize,RT_LIB),Cint,()))
 end
 function capture!(art,a)
  h=similar(a.host);GC.@preserve h rtck(ccall((:cudaMemcpy,RT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),pointer(h),a.base,a.bytes,2))
  if a.bf;append!(art,decode_bf.(h));else;append!(art,h);end
 end
 for c in cases
  println("WARM_PATH kind=",c.code," M=",c.m," N=",c.n," K=",c.k);prefetch(c);pipeline(c,handle[],workspace)
 end
 for trial in 1:5
  art=copy(pre);times=Float64[]
  for c in cases
   GC.@preserve c rtck(ccall((:cudaMemcpy,RT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),c.C.base,pointer(c.C.host),c.C.bytes,1));capture!(art,c.C);prefetch(c)
   rtck(ccall((:cudaDeviceSynchronize,RT_LIB),Cint,()));t=time_ns();pipeline(c,handle[],workspace);rtck(ccall((:cudaDeviceSynchronize,RT_LIB),Cint,()));push!(times,(time_ns()-t)*1e-9)
   for a in (c.C,c.A,c.B);capture!(art,a);end
  end
  wg=zeros(UInt8,512);GC.@preserve wg begin
   rtck(ccall((:cudaMemcpy,RT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),pointer(wg),wb[],256,2));rtck(ccall((:cudaMemcpy,RT_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),pointer(wg,257),wb[]+256+33554432,256,2))
  end
  append!(art,wg);@assert length(art)==length(pre)+length(gold);actual=art[length(pre)+1:end];ok=all(isfinite,actual)&&actual==gold;ma=maximum(abs.(actual.-gold));path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum27_source_cublasLt_descriptor_heuristic_matmul_sync_destroy_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"authored_device_kernels\":0,\"workspace_bytes\":33554432,\"managed_allocations\":true,\"FP32_actual_compute_type_all_source_flags\":68}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,path,length(art),string(times))
 end
 for c in cases,a in (c.A,c.B,c.C);rtck(ccall((:cudaFree,RT_LIB),Cint,(Ptr{Cvoid},),a.base));end
 rtck(ccall((:cudaFree,RT_LIB),Cint,(Ptr{Cvoid},),wb[]));ltck(ccall((:cublasLtDestroy,LT_LIB),Cint,(Ptr{Cvoid},),handle[]))
end
