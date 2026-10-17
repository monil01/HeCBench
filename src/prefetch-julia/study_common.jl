
using CUDA,Printf
const PREFETCH_NS=(4099,1048579)
const PREFETCH_REPEATS=7
function canonical(n)
 a=Vector{Float32}(undef,n);b=similar(a)
 for i=0:n-1
  a[i+1]=n==4099 ? 1f0 : Float32(mod(i,17)-8)/16f0
  b[i+1]=n==4099 ? 2f0 : Float32(mod(i*5,23)-11)/32f0
 end
 a,b
end
function prefetch_study(launch!)
 @assert CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)==1
 CUDA.stream!(CUDA.default_stream());states=[];prefix=Float32[]
 for n in PREFETCH_NS
  ca,cb=canonical(n);append!(prefix,Float32[n,PREFETCH_REPEATS]);append!(prefix,ca);append!(prefix,cb)
  for mode in (true,false)
   a=CuArray{Float32,1,CUDA.UnifiedMemory}(undef,n);b=similar(a)
   ha=unsafe_wrap(Array,convert(Ptr{Float32},a.data[].mem),(n,);own=false);hb=unsafe_wrap(Array,convert(Ptr{Float32},b.data[].mem),(n,);own=false)
   # Direct device views bypass installed CUDA.jl KernelAdaptor automatic prefetch.
   # The CuArray owners and host views remain alive in states through every sync.
   da=Base.unsafe_convert(CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global},a);db=Base.unsafe_convert(CUDA.CuDeviceArray{Float32,1,CUDA.AS.Global},b)
   copyto!(ha,ca);copyto!(hb,cb);launch!(Int32(n),da,db)
   if mode;CUDA.prefetch(a.data[].mem;stream=CUDA.default_stream());CUDA.prefetch(b.data[].mem;stream=CUDA.default_stream());end
   CUDA.device_synchronize(blocking=true)
   push!(states,(n,mode,a,b,ha,hb,da,db,ca,cb))
  end
 end
 CUDA.device_synchronize(blocking=true);allok=true
 for trial=1:5
  # CPU writes every managed page after warmup/last capture; setup outside ROI.
  for (n,mode,a,b,ha,hb,da,db,ca,cb) in states;copyto!(ha,ca);copyto!(hb,cb);end
  CUDA.device_synchronize(blocking=true);times=Float64[]
  for (n,mode,a,b,ha,hb,da,db,ca,cb) in states
   start=time_ns()
   for _=1:PREFETCH_REPEATS
    if mode
     CUDA.prefetch(a.data[].mem;stream=CUDA.default_stream());CUDA.prefetch(b.data[].mem;stream=CUDA.default_stream())
    end
    launch!(Int32(n),da,db);CUDA.device_synchronize(blocking=true)
   end
   push!(times,(time_ns()-start)*1e-9/PREFETCH_REPEATS)
  end
  out=copy(prefix);ok=true
  for (n,mode,a,b,ha,hb,da,db,ca,cb) in states
   ok&=ha==ca
   for i=1:n;ok&=hb[i]==cb[i]+Float32(PREFETCH_REPEATS)*ca[i];end
   append!(out,ha);append!(out,hb)
  end
  allok&=ok;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":4210712,\"max_abs_error\":%d,\"max_rel_error\":%d,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":7,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,ok ? "true" : "false",ok ? 0 : 1,ok ? 0 : 1,path,length(out))
 end
 allok||error("Managedmemory fulloutput mismatch");println("PASS")
end
