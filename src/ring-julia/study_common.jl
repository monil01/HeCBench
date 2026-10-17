const RING_CUDART=CUDA.CUDACore.CUDA_Runtime.libcudart
runtime_setdevice(d)=((status=ccall((:cudaSetDevice,RING_CUDART),Cint,(Cint,),d));status==0||error("cudaSetDevice failed $status");nothing)
function sync_both!()
 for d in 0:1
  runtime_setdevice(d)
  status=ccall((:cudaDeviceSynchronize,RING_CUDART),Cint,())
  status==0||error("cudaDeviceSynchronize failed $status")
 end
 return
end
function exchange!(buffers,n,repeat)
 GC.@preserve buffers begin
  p0=Ptr{Cvoid}(UInt(pointer(buffers[1])));p1=Ptr{Cvoid}(UInt(pointer(buffers[2])))
  for _ in 1:repeat
   for (dst,src) in ((p1,p0),(p0,p1))
    status=ccall((:cudaMemcpy,RING_CUDART),Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t,Cint),dst,src,4n,4)
    status==0||error("cudaMemcpyDefault failed $status")
   end
  end
 end
 sync_both!()
 return
end
function study(allocate)
 devices=collect(CUDA.devices());length(devices)==2||error("Expected exactly2 visibledevices")
 n=65536;repeat=10;input=collect(Int32(0):Int32(n-1));buffers=Any[]
 for dev in devices
  CUDA.device!(dev) do
   push!(buffers,allocate(n))
  end
 end
 function reset!()
  for (dev,array) in zip(devices,buffers)
   CUDA.device!(dev) do;fill!(array,Int32(0));end
  end
  CUDA.device!(devices[1]) do;copyto!(buffers[1],input);end
  sync_both!()
 end
 reset!();exchange!(buffers,n,repeat)
 for trial in 1:5
  ring_guard();reset!();start=time_ns();exchange!(buffers,n,repeat);seconds=(time_ns()-start)*1e-9/repeat
  output=Int32[];correct=true
  for (dev,array) in zip(devices,buffers)
   CUDA.device!(dev) do
    got=Array(array);correct &=got==input;append!(output,got)
   end
  end
  ring_guard();path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,output),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",2n,",\"max_abs_error\":",correct ? 0 : 1,",\"timing_region\":\"mean_two_device_ring_round_with_final_both_device_fences\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"devices_checked\":2,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",2n,",\"byte_order\":\"little\"}}")
  correct||error("Full buffers on both devices differ")
 end
 version=Ref{Cint}();ccall((:cudaRuntimeGetVersion,RING_CUDART),Cint,(Ref{Cint},),version)
 println("STUDY_CUDART ",realpath(Libdl.dlpath(RING_CUDART))," version=",version[])
 for (dev,array) in zip(devices,buffers)
  CUDA.device!(dev) do;CUDA.unsafe_free!(array);end
 end
 return
end
