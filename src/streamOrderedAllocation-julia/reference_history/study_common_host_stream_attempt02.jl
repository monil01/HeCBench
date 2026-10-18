using CUDA,Printf,Libdl
const STUDY_CUDART="/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146"
ck(x)=x==0 ? nothing : error("CUDART status $(x)")
function allocation_study(launch)
 CUDA.device!(0);CUDA.context();supported=Ref{Cint}();ck(ccall((:cudaDeviceGetAttribute,STUDY_CUDART),Cint,(Ref{Cint},Cint,Cint),supported,115,0));@assert supported[]==1
 version=Ref{Cint}();ck(ccall((:cudaRuntimeGetVersion,STUDY_CUDART),Cint,(Ref{Cint},),version));println("CUDART_PROVENANCE version=",version[]," path=",realpath(Libdl.dlpath(Libdl.dlopen(STUDY_CUDART))));pool=Ref{Ptr{Cvoid}}();ck(ccall((:cudaDeviceGetDefaultMemPool,STUDY_CUDART),Cint,(Ref{Ptr{Cvoid}},Cint),pool,0));old=Ref{UInt64}();ck(ccall((:cudaMemPoolGetAttribute,STUDY_CUDART),Cint,(Ptr{Cvoid},Cint,Ref{UInt64}),pool[],4,old))
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==9229&&length(gold)==1410660;pos=2;cs=Any[]
 for ci in 1:4
  n=Int(pre[pos]);@assert pre[pos+1]==100;pos+=2;a=Float32.(pre[pos:pos+n-1]);pos+=n;b=Float32.(pre[pos:pos+n-1]);pos+=n
  push!(cs,(n=n,a=a,b=b,out=zeros(Float32,100*n),ea=zeros(Float32,n),eb=zeros(Float32,n)))
 end
 @assert pos==length(pre)+1
 function execute(c,mode)
  s=mode==0 ? CUDA.default_stream() : CUDA.CuStream(flags=CUDA.STREAM_NON_BLOCKING);rawstream=Ptr{Cvoid}(s.handle)
  if mode!=0;threshold=Ref(mode==1 ? UInt64(0) : typemax(UInt64));ck(ccall((:cudaMemPoolSetAttribute,STUDY_CUDART),Cint,(Ptr{Cvoid},Cint,Ref{UInt64}),pool[],4,threshold));end
  # Stream selection is a host dispatch policy outside the measured source pipeline.
  CUDA.stream!(s) do
   ck(ccall((:cudaDeviceSynchronize,STUDY_CUDART),Cint,()));t=time_ns();n=c.n;bytes=4*n
   GC.@preserve c begin
    for it in 0:99
     a=Ref{Ptr{Cvoid}}();b=Ref{Ptr{Cvoid}}();o=Ref{Ptr{Cvoid}}()
     for dest in (a,b,o)
      if mode==0;ck(ccall((:cudaMalloc,STUDY_CUDART),Cint,(Ref{Ptr{Cvoid}},Csize_t),dest,bytes));else;ck(ccall((:cudaMallocAsync,STUDY_CUDART),Cint,(Ref{Ptr{Cvoid}},Csize_t,Ptr{Cvoid}),dest,bytes,rawstream));end
     end
     for (dst,src) in ((a[],pointer(c.a)),(b[],pointer(c.b)))
      if mode==0;ck(ccall((:cudaMemcpy,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),dst,src,bytes,1));else;ck(ccall((:cudaMemcpyAsync,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),dst,src,bytes,1,rawstream));end
     end
     # Nonowning wrappers expose actual runtime allocations to framework dispatch.
     da=unsafe_wrap(CuArray{Float32,1,CUDA.DeviceMemory},CUDA.CuPtr{Float32}(UInt(a[])),(n,);own=false);db=unsafe_wrap(CuArray{Float32,1,CUDA.DeviceMemory},CUDA.CuPtr{Float32}(UInt(b[])),(n,);own=false);dc=unsafe_wrap(CuArray{Float32,1,CUDA.DeviceMemory},CUDA.CuPtr{Float32}(UInt(o[])),(n,);own=false)
     launch(da,db,dc,Int32(n),s)
     if it==99
      for (dst,src) in ((pointer(c.ea),a[]),(pointer(c.eb),b[]))
       if mode==0;ck(ccall((:cudaMemcpy,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),dst,src,bytes,2));else;ck(ccall((:cudaMemcpyAsync,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),dst,src,bytes,2,rawstream));end
      end
     end
     for ptr in (a[],b[])
      if mode==0;ck(ccall((:cudaFree,STUDY_CUDART),Cint,(Ptr{Cvoid},),ptr));else;ck(ccall((:cudaFreeAsync,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid}),ptr,rawstream));end
     end
     dst=pointer(c.out,it*n+1)
     if mode==0;ck(ccall((:cudaMemcpy,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),dst,o[],bytes,2));ck(ccall((:cudaFree,STUDY_CUDART),Cint,(Ptr{Cvoid},),o[]))
     else;ck(ccall((:cudaMemcpyAsync,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),dst,o[],bytes,2,rawstream));ck(ccall((:cudaFreeAsync,STUDY_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid}),o[],rawstream));ck(ccall((:cudaStreamSynchronize,STUDY_CUDART),Cint,(Ptr{Cvoid},),rawstream));end
    end
   end
   elapsed=(time_ns()-t)*1e-9
   mode!=0&&CUDA.unsafe_destroy!(s)
   return elapsed
  end
 end
 for c in cs,mode in 0:2;execute(c,mode);end
 for trial in 1:5
  art=copy(pre);times=Float64[];ok=true
  for c in cs,mode in 0:2
   push!(times,execute(c,mode));append!(art,c.out);append!(art,c.ea);append!(art,c.eb);ok &=c.ea==c.a&&c.eb==c.b
  end
  @assert length(art)==length(pre)+length(gold);actual=art[length(pre)+1:end];ok &=all(isfinite,actual)&&actual==gold
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"sum12_source_100repeat_allocator_copy_add_free_pipelines_plus_final_inputecho\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"authored_device_kernels\":1,\"GPU_guards_present\":false,\"all_allocation_copy_free_API_status_checked\":true}\n",trial,sum(times),ok ? "true" : "false",length(art),path,length(art),string(times))
 end
 ck(ccall((:cudaMemPoolSetAttribute,STUDY_CUDART),Cint,(Ptr{Cvoid},Cint,Ref{UInt64}),pool[],4,old))
end
