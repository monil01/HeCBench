const CCD_RT=CUDA.CUDACore.CUDA_Runtime.libcudart
const CCD_N=4096
const CCD_G=64
const CCD_SENT=-777f0
raw(x::Float32)=reinterpret(UInt32,x)
ck(e)=e==0 ? nothing : error("CUDART status $e")
function pipeline(c)
 copies=UInt32(0);creates=UInt32(0);destroys=UInt32(0);syncs=UInt32(0)
 GC.@preserve c begin
  for iter in 1:c.it
   if c.mode==1
    for s in 1:c.s
     ck(ccall((:cudaStreamCreate,CCD_RT),Cint,(Ptr{Ptr{Cvoid}},),pointer(c.streams,s)));creates+=1
    end
    for stream in c.streams,d in c.ptrs
     ck(ccall((:cudaMemcpyAsync,CCD_RT),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),d,c.hptr,CCD_N*4,1,stream));copies+=1
    end
    for stream in c.streams
     ck(ccall((:cudaStreamSynchronize,CCD_RT),Cint,(Ptr{Cvoid},),stream));syncs+=1
     ck(ccall((:cudaStreamDestroy,CCD_RT),Cint,(Ptr{Cvoid},),stream));destroys+=1
    end
   else
    for s in 1:c.s,d in c.ptrs
     ck(ccall((:cudaMemcpyAsync,CCD_RT),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint,Ptr{Cvoid}),d,c.hptr,CCD_N*4,1,C_NULL));copies+=1
    end
    ck(ccall((:cudaDeviceSynchronize,CCD_RT),Cint,()));syncs+=1
   end
  end
 end
 (copies,creates,destroys,syncs)
end
function reset(c)
 for ptr in c.ptrs;ck(ccall((:cudaMemset,CCD_RT),Cint,(Ptr{Cvoid},Cint,Csize_t),ptr,255,CCD_N*4));end
 ck(ccall((:cudaDeviceSynchronize,CCD_RT),Cint,()))
end
function study(allocate)
 oracle=open(io->read!(io,Vector{UInt32}(undef,CCD_N)),joinpath(@__DIR__,"oracle_u32.bin"));filesize(joinpath(@__DIR__,"oracle_u32.bin"))==CCD_N*4||error("oracle size")
 cases=Any[]
 for mode in 0:1,test in (0,5,10,15)
  group=test÷4;s=(1,2,4,8)[test%4+1];b=(1,100,1000,5000)[group+1];it=256÷(s*(1<<(group+1)));h=fill(CCD_SENT,CCD_N+2CCD_G)
  for i in 0:CCD_N-1;h[CCD_G+i+1]=1.618f0+Float32(i);raw(h[CCD_G+i+1])==oracle[i+1]||error("input oracle");end
  ds=[allocate(CCD_N+2CCD_G)for _ in 1:b];CUDA.device_synchronize(blocking=true)
  ptrs=Ptr{Cvoid}[];baseptrs=Ptr{Cvoid}[]
  for d in ds
   base=Ptr{Cvoid}(UInt(pointer(d)));push!(baseptrs,base);push!(ptrs,base+4CCD_G)
   GC.@preserve h d ck(ccall((:cudaMemcpy,CCD_RT),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),base,pointer(h),length(h)*4,1))
  end
  push!(cases,(mode=mode,test=test,s=s,b=b,it=it,h=h,ds=ds,ptrs=ptrs,baseptrs=baseptrs,hptr=Ptr{Cvoid}(pointer(h,CCD_G+1)),streams=Vector{Ptr{Cvoid}}(undef,s)))
 end
 for c in cases;reset(c);pipeline(c);end
 for trial in 1:5
  out=UInt32[8];sizehint!(out,51574089);times=Float64[];ok=true
  for c in cases
   append!(out,UInt32[c.test,c.mode,c.s,c.b,c.it]);append!(out,reinterpret(UInt32,view(c.h,CCD_G+1:CCD_G+CCD_N)))
  end
  for c in cases
   reset(c);start=time_ns();calls=pipeline(c);dt=(time_ns()-start)*1e-9;push!(times,dt)
   h=zeros(Float32,CCD_N+2CCD_G)
   for ptr in c.baseptrs
    GC.@preserve h c ck(ccall((:cudaMemcpy,CCD_RT),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),pointer(h),ptr,length(h)*4,2))
    for i in eachindex(h)
     v=raw(h[i]);ok &=v==(i<=CCD_G||i>CCD_G+CCD_N ? raw(CCD_SENT) : oracle[i-CCD_G]);push!(out,v)
    end
   end
   ok &=reinterpret(UInt32,view(c.h,CCD_G+1:CCD_G+CCD_N))==oracle
   ok &=all(==(CCD_SENT),view(c.h,1:CCD_G))&&all(==(CCD_SENT),view(c.h,CCD_G+CCD_N+1:length(c.h)))
   expected=(UInt32(c.it*c.s*c.b),UInt32(c.mode==1 ? c.it*c.s : 0),UInt32(c.mode==1 ? c.it*c.s : 0),UInt32(c.mode==1 ? c.it*c.s : c.it));ok &=calls==expected;append!(out,calls)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum8_pageable_H2D_stream_create_sync_destroy_runtime_pipeline\",\"jit_excluded\":true,\"internal_repeats\":1,\"authored_device_kernel_calls\":0,\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("copy input output guards/APIcounts mismatch")
 end
 CUDA.device_synchronize(blocking=true)
 for c in cases,d in c.ds;CUDA.unsafe_free!(d);end
end
