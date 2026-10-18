const SMD_RT=CUDA.CUDACore.CUDA_Runtime.libcudart
const SMD_G=64
const SMD_PART=8192
const SMD_CASES=((3,1),(8193,1),(131079,1),(33554432,0))
const SMD_SENT=-777f0
u32(x::Float32)=UInt64(reinterpret(UInt32,x))
u64(x::Float64)=reinterpret(UInt64,x)
checkrt(s)=s==0 ? nothing : error("CUDART status $s")
function input_value(i,flavor)
 flavor==0&&return Float32(i&255)/512f0
 x=UInt32(i)*UInt32(1664525)+UInt32(1013904223);x⊻=x>>16;x*=UInt32(2246822519);x⊻=x>>13
 Float32(x)*Float32(2.0^-32)
end
function pinned(n)
 p=Ref{Ptr{Float32}}();checkrt(ccall((:cudaMallocHost,SMD_RT),Cint,(Ref{Ptr{Float32}},Csize_t),p,4n));unsafe_wrap(Array,p[],(n,);own=false)
end
function loadoracle(g)
 p=joinpath(@__DIR__,"oracle_g$(g)_u64.bin");filesize(p)%8==0||error("oracle size")
 open(io->read!(io,Vector{UInt64}(undef,filesize(p)÷8)),p)
end
function capture_plan!(payload::Vector{UInt64},suffix::Vector{UInt64},p,flavor::Int,cpu::Float64)
 CUDA.device!(p.dev) do
  unsafe_copyto!(pointer(p.checkin),pointer(p.din),length(p.checkin);stream=p.stream)
  unsafe_copyto!(pointer(p.checkout),pointer(p.dout),length(p.checkout);stream=p.stream)
 end
 ok=true
 for i in 1:p.n
  v=p.checkin[SMD_G+i];ok &=reinterpret(UInt32,v)==reinterpret(UInt32,p.hin[SMD_G+i]) && v==input_value(p.offset+i-1,flavor);push!(payload,u32(v));cpu+=Float64(p.hin[SMD_G+i])
 end
 local_sum=0f0
 for i in 1:SMD_PART
  v=p.hout[SMD_G+i];push!(suffix,u32(v));local_sum+=v
 end
 for i in 1:SMD_PART;push!(suffix,u32(p.checkout[SMD_G+i]));end
 push!(suffix,u32(local_sum))
 for a in(p.hin,p.checkin,p.hout,p.checkout)
  for v in view(a,1:SMD_G);ok &=v==SMD_SENT;push!(suffix,u32(v));end
  for v in view(a,length(a)-SMD_G+1:length(a));ok &=v==SMD_SENT;push!(suffix,u32(v));end
 end
 ok,cpu,local_sum
end
function study(allocate,launch)
 g=parse(Int,ARGS[1]);devices=collect(CUDA.devices());length(devices)==g||error("Actualvisibledevicecount differs")
 for (d,dev) in enumerate(devices)
  println("STUDY_DEVICE visible=",d-1," uuid=",CUDA.uuid(dev))
 end
 oracle=loadoracle(g);prefixcount=1+sum(2+n for(n,_)in SMD_CASES);payload=UInt64[];sizehint!(payload,prefixcount+length(oracle));cases=Any[]
 for(n,flavor)in SMD_CASES
  plans=Any[];offset=0
  for (d,dev) in enumerate(devices)
   count=n÷g+(d<=n%g ? 1 : 0);source_offset=offset::Int
   plan=CUDA.device!(dev) do
    stream=CUDA.CuStream(flags=CUDA.STREAM_DEFAULT)
    hin=pinned(count+2SMD_G);hout=pinned(SMD_PART+2SMD_G);checkin=pinned(count+2SMD_G);checkout=pinned(SMD_PART+2SMD_G)
    fill!(hin,SMD_SENT);fill!(hout,SMD_SENT)
    for i in 0:count-1;hin[SMD_G+i+1]=input_value(source_offset+i,flavor);end
    din=allocate(count+2SMD_G);dout=allocate(SMD_PART+2SMD_G)
    CUDA.device_synchronize(blocking=true) # allocation/initialization completion before foreign-stream raw copies
    CUDA.stream!(stream) do
     unsafe_copyto!(pointer(din),pointer(hin),length(hin);stream=stream,async=true)
     unsafe_copyto!(pointer(dout),pointer(hout),length(hout);stream=stream,async=true)
    end
    CUDA.synchronize(stream;blocking=true)
    (dev=dev,stream=stream,n=count,offset=offset,hin=hin,hout=hout,checkin=checkin,checkout=checkout,din=din,dout=dout,input=view(din,SMD_G+1:SMD_G+count),output=view(dout,SMD_G+1:SMD_G+SMD_PART))
   end
   push!(plans,plan);offset+=count
  end
  push!(cases,(n=n,flavor=flavor,plans=plans))
 end
 function poison(c)
  for p in c.plans
   CUDA.device!(p.dev) do
    fill!(reinterpret(UInt32,view(p.hout,SMD_G+1:SMD_G+SMD_PART)),UInt32(0xffffffff))
    stream=Ptr{Cvoid}(p.stream.handle)
    checkrt(ccall((:cudaMemsetAsync,SMD_RT),Cint,(Ptr{Cvoid},Cint,Csize_t,Ptr{Cvoid}),Ptr{Cvoid}(UInt(pointer(p.input))),255,p.n*4,stream))
    checkrt(ccall((:cudaMemsetAsync,SMD_RT),Cint,(Ptr{Cvoid},Cint,Csize_t,Ptr{Cvoid}),Ptr{Cvoid}(UInt(pointer(p.output))),255,SMD_PART*4,stream))
    CUDA.synchronize(p.stream;blocking=true)
   end
  end
 end
 function pipeline(c)
  for p in c.plans
   CUDA.device!(p.dev) do
    # Managed pointer conversion must see the owning stream for the entire submission.
    # Otherwise conversion of p.output on the restored task stream waits for this
    # device's kernel before the next device has received its submissions.
    CUDA.stream!(p.stream) do
     unsafe_copyto!(pointer(p.input),pointer(p.hin,SMD_G+1),p.n;stream=p.stream,async=true)
     launch(p.output,p.input,p.n)
     unsafe_copyto!(pointer(p.hout,SMD_G+1),pointer(p.output),SMD_PART;stream=p.stream,async=true)
    end
   end
  end
  for p in c.plans
   CUDA.device!(p.dev) do;CUDA.synchronize(p.stream;blocking=true);end
  end
 end
 for c in cases;poison(c);pipeline(c);end
 for trial in 1:5
  study_guard();empty!(payload);push!(payload,u32(Float32(length(cases))));suffix=UInt64[];sizehint!(suffix,length(oracle));total=0.;times=Float64[];ok=true
  for c in cases
   poison(c);start=time_ns();pipeline(c);elapsed=(time_ns()-start)*1e-9;total+=elapsed;push!(times,elapsed)
   push!(payload,u32(Float32(c.n)),u32(Float32(c.flavor)));locals=Float32[];cpu=0.
   for p in c.plans
    valid,cpu,local_sum=capture_plan!(payload,suffix,p,c.flavor,cpu)
    ok &=valid;push!(locals,local_sum)
   end
   gpu=0f0;for v in locals;gpu+=v;end
   diff=abs(cpu-Float64(gpu))/abs(cpu);ok &=isfinite(diff)&&diff<1e-5;append!(suffix,UInt64[u32(gpu),u64(cpu),u64(diff)])
  end
  ok &=suffix==oracle;append!(payload,suffix);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");study_guard()
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum4_async_pinned_H2D_stridedpartials_D2H_all_stream_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":",4g,",\"active_devices\":",g,",\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("Fullpartialvectors/hostfolds/CPUreference/input/guards mismatch")
 end
 for c in cases,p in c.plans
  CUDA.device!(p.dev) do
   CUDA.synchronize(p.stream;blocking=true)
   for a in(p.hin,p.hout,p.checkin,p.checkout);checkrt(ccall((:cudaFreeHost,SMD_RT),Cint,(Ptr{Cvoid},),pointer(a)));end
   Base.finalize(p.stream)
  end
 end
end
