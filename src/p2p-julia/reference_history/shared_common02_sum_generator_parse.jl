const P2P_CUDART=CUDA.CUDACore.CUDA_Runtime.libcudart
const P2P_SENT=-12345f0
const P2P_COPIES=10
ck(x)=x==0 ? nothing : error("CUDART status $x")
function syncboth(devices)
 for dev in devices
  CUDA.device!(dev) do
   ck(ccall((:cudaDeviceSynchronize,P2P_CUDART),Cint,()))
  end
 end
end
function study(allocate,launch)
 devices=collect(CUDA.devices());length(devices)==2||error("Exactlytwo authorized GPUs needed")
 for d in 1:2
  cap=Ref{Cint}();ck(ccall((:cudaDeviceCanAccessPeer,P2P_CUDART),Cint,(Ref{Cint},Cint,Cint),cap,d-1,2-d));cap[]==1||error("No actual peer capability")
  CUDA.CUDACore.maybe_enable_peer_access(devices[d],devices[3-d])==1||error("Peer mapping enable failed")
  println("STUDY_PEER visible=",d-1," peer=",2-d," uuid=",CUDA.uuid(devices[d])," enabled=true")
 end
 cases=Any[]
 for n in (256,65536,16777216)
  seed=Float32[Float32(i) for i in 0:4095];input=repeat(seed,cld(n,4096))[1:n];buffers=Any[];ptrs=Ptr{Cvoid}[]
  for dev in devices
   CUDA.device!(dev) do
    a=allocate(n+2);push!(buffers,a);push!(ptrs,Ptr{Cvoid}(UInt(pointer(a,2))))
   end
  end
  push!(cases,(n=n,input=input,buffers=buffers,ptrs=ptrs))
 end
 function reset(c)
  for d in 1:2
   host=fill(P2P_SENT,c.n+2);host[2:end-1].=d==1 ? c.input : -3f0
   CUDA.device!(devices[d]) do
    copyto!(c.buffers[d],host)
   end
  end
  syncboth(devices)
 end
 function exchange(c)
  CUDA.device!(devices[2]) do
   for i in 0:P2P_COPIES-1
    ck(ccall((:cudaMemcpy,P2P_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),c.ptrs[(i+1)%2+1],c.ptrs[i%2+1],c.n*4,4))
   end
   syncboth(devices)
  end
 end
 function remote(c)
  CUDA.device!(devices[1]) do
   copyto!(c.buffers[1],2,c.input,1,c.n);ck(ccall((:cudaDeviceSynchronize,P2P_CUDART),Cint,()))
  end
  for (target,source)in ((2,1),(1,2))
   CUDA.device!(devices[target]) do
    launch(c.n,c.buffers[source],c.buffers[target]);ck(ccall((:cudaDeviceSynchronize,P2P_CUDART),Cint,()))
   end
  end
 end
 function capture(c,io,doubled)
  ok=true
  for d in 1:2
   host=CUDA.device!(devices[d]) do
    Array(c.buffers[d])
   end
   ok&=host[1]==P2P_SENT&&host[end]==P2P_SENT
   scale=doubled ? (d==1 ? 4f0 : 2f0) : 1f0
   for i in 1:c.n
    ok&=host[i+1]==c.input[i]*scale
   end
   write(io,host)
  end
  ok
 end
 for c in cases
  reset(c);exchange(c);remote(c)
 end
 count=1+sum(c.n+2+4*(c.n+2)for c in cases)
 for trial in 1:5
  p2p_guard();path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");total=0.;times=Float64[];ok=true
  open(path,"w")do io
   write(io,3f0)
   for c in cases
    write(io,Float32[c.n,P2P_COPIES]);write(io,c.input)
   end
   for c in cases
    reset(c);start=time_ns();exchange(c);t=(time_ns()-start)*1e-9/P2P_COPIES;total+=t;push!(times,t)
    ok&=capture(c,io,false);remote(c);ok&=capture(c,io,true)
   end
  end
  p2p_guard()
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",count,",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum3_case_mean10_alternating_UVA_copies_both_device_completion\",\"jit_excluded\":true,\"internal_repeats\":10,\"timed_kernel_calls\":0,\"untimed_remote_kernel_calls\":6,\"case_percopy_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",count,",\"byte_order\":\"little\"}}")
  ok||error("Both complete copy/remote outputs or guards differ")
 end
 for c in cases,d in 1:2
  CUDA.device!(devices[d]) do
   CUDA.unsafe_free!(c.buffers[d])
  end
 end
end
