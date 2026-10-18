using CUDA,Printf
const PCC_CUBLAS="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
include("vendor_probe.jl")
function pcc_study(extract)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));pos=2;cs=Any[]
 for ci in 1:5
  N,L,avail=Int.(pre[pos:pos+2]);pos+=3;raw=Float32.(pre[pos:pos+N*L-1]);pos+=N*L;norm=Float32.(pre[pos:pos+N*L-1]);pos+=N*L;h=vcat(fill(-12345f0,64),norm,fill(-12345f0,8));d=CuArray(h);chunks=Any[];off=0;zero=[all(iszero,norm[i*L+1:(i+1)*L]) for i in 0:N-1]
  while off<N
   n=N-off;n1=avail==0 ? n : min(n,avail÷(2*n));@assert n1>0;nc=n1*n;nu=n1*n-n1*(n1+1)÷2
   co=CuArray(vcat(fill(-12345f0,64),fill(Float32(NaN),nc),fill(-12345f0,8)));uo=CuArray(vcat(fill(-12345f0,64),fill(Float32(NaN),nu),fill(-12345f0,8)))
   push!(chunks,(off=off,n=n,n1=n1,nc=nc,nu=nu,co=co,uo=uo,C=view(co,65:nc+64),U=view(uo,65:nu+64)));off+=n1
  end
  push!(cs,(N=N,L=L,avail=avail,raw=raw,h=h,d=d,zero=zero,chunks=chunks))
 end
 @assert pos==length(pre)+1;handle=make_study_cublas();alpha=Ref(1f0);beta=Ref(0f0)
 function gemm(c,ch)
  ptr=pointer(c.d,65+ch.off*c.L)
  status=ccall((:cublasSgemm_v2,PCC_CUBLAS),Cint,(CUDA.CUBLAS.cublasHandle_t,Cint,Cint,Cint,Cint,Cint,Ptr{Float32},CUDA.CuPtr{Float32},Cint,CUDA.CuPtr{Float32},Cint,Ptr{Float32},CUDA.CuPtr{Float32},Cint),handle,Cint(CUDA.CUBLAS.CUBLAS_OP_T),Cint(CUDA.CUBLAS.CUBLAS_OP_N),ch.n1,ch.n,c.L,alpha,ptr,c.L,ptr,c.L,beta,pointer(ch.co,65),ch.n1)
  @assert status==0
 end
 for c in cs,ch in c.chunks;gemm(c,ch);CUDA.synchronize();extract(c,ch);CUDA.synchronize();end
 for trial in 1:5
  art=copy(pre);go=0;ok=true;ma=0.;mr=0.;times=Float64[];stages=Vector{Float64}[]
  for c in cs
   sumcase=0.
   for ch in c.chunks
    fill!(ch.C,Float32(NaN));fill!(ch.U,Float32(NaN));CUDA.synchronize();t=time_ns();gemm(c,ch);CUDA.synchronize();gt=(time_ns()-t)*1e-9;t=time_ns();extract(c,ch);CUDA.synchronize();et=(time_ns()-t)*1e-9;push!(stages,[gt,et]);sumcase+=gt+et
    for (kind,d) in enumerate([ch.co,ch.uo])
     v=Float64.(Array(d));ref=gold[go+1:go+length(v)];go+=length(v);ok&=all(isfinite,v)&&v[1:64]==ref[1:64]&&v[end-7:end]==ref[end-7:end];er=abs.(v.-ref);ok&=all(er .<= 1e-6 .+ 3e-6 .* abs.(ref));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(ref).+1e-300)))
     if kind==1
      for j in 0:ch.n-1,i in 0:ch.n1-1
       x=v[65+j*ch.n1+i];ok&=abs(x)<=1+3e-6
       if c.zero[ch.off+i+1]||c.zero[ch.off+j+1];ok&=x==0;end
       if i==j&&!c.zero[ch.off+i+1];ok&=abs(x-1)<=3e-6;end
      end
     else
      posu=65
      for i in 0:ch.n1-1,j in i+1:ch.n-1
       x=v[posu];posu+=1;ok&=abs(x)<=1+3e-6
       if c.zero[ch.off+i+1]||c.zero[ch.off+j+1];ok&=x==0;end
      end
     end
     append!(art,v)
    end
   end
   push!(times,sumcase);h=Array(c.d);ok&=reinterpret(UInt8,h)==reinterpret(UInt8,c.h);v=Float64.(h);ref=gold[go+1:go+length(v)];go+=length(v);ok&=v==ref;append!(art,v)
  end
  @assert go==length(gold);f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum5_source_cublas_GEMM_plus_single_multi_upper_extraction_components\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"chunk_GEMM_extraction_stage_times_seconds\":%s,\"authored_GEMM_kernels\":0}\n",trial,sum(times),ok ? "true" : "false",go,ma,mr,f,length(art),string(times),string(stages))
 end
 CUDA.CUBLAS.cublasDestroy_v2(handle)
end
