using CUDA,Printf
const STREAMUM_NS=(64,99,100,101,257,511,67,127)
function streamum_values(n,t)
 a=[Float64(mod(13i+7j+3t,31)-15)/32 for j=0:n-1 for i=0:n-1]
 x=[Float64(mod(5j+3t,17)-8)/16 for j=0:n-1]
 a,x
end
function streamum_gold(a,x,n)
 # Exact dyadic arithmetic: integer numerator sum /512, independently indexed.
 [sum(Int(round(a[(n<100 ? i*n+j : j*n+i)+1]*32))*Int(round(x[j+1]*16)) for j=0:n-1)/512 for i=0:n-1]
end
function streamum_study(launch!)
 @assert Threads.nthreads()>=4
 @assert CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)==1
 states=[];prefix=Float64[];streams=[CuStream() for _=1:4]
 for (t,n) in enumerate(STREAMUM_NS)
  ca,cx=streamum_values(n,t-1)
  a=CuArray{Float64,1,CUDA.UnifiedMemory}(undef,n*n);x=similar(a,n);r=similar(a,n)
  ha=unsafe_wrap(Array,convert(Ptr{Float64},a.data[].mem),(n*n,);own=false)
  hx=unsafe_wrap(Array,convert(Ptr{Float64},x.data[].mem),(n,);own=false)
  hr=unsafe_wrap(Array,convert(Ptr{Float64},r.data[].mem),(n,);own=false)
  da=Base.unsafe_convert(CUDA.CuDeviceArray{Float64,1,CUDA.AS.Global},a)
  dx=Base.unsafe_convert(CUDA.CuDeviceArray{Float64,1,CUDA.AS.Global},x)
  dr=Base.unsafe_convert(CUDA.CuDeviceArray{Float64,1,CUDA.AS.Global},r)
  append!(prefix,Float64[n,t-1,n<100 ? 0 : 1]);append!(prefix,ca);append!(prefix,cx);append!(prefix,zeros(n))
  push!(states,(n,a,x,r,ha,hx,hr,da,dx,dr,ca,cx,streamum_gold(ca,cx,n)))
 end
 function reset!()
  for (n,a,x,r,ha,hx,hr,da,dx,dr,ca,cx,gold) in states
   copyto!(ha,ca);copyto!(hx,cx);fill!(hr,0)
  end
 end
 next=Threads.Atomic{Int}(1);assignments=zeros(Int,length(states))
 function prepare_dispatch!()
  next[]=1;fill!(assignments,-1)
 end
 function execute_all!()
  @sync for w=1:4
   Threads.@spawn begin
    CUDA.device!(0)
    CUDA.stream!(streams[w]) do
     while true
      k=Threads.atomic_add!(next,1);k>length(states)&&break
      n,a,x,r,ha,hx,hr,da,dx,dr,ca,cx,gold=states[k];assignments[k]=w-1
      if n<100
       gemv_host!(hr,ha,hx,n)
      else
       launch!(n,dr,da,dx)
      end
     end
    end
   end
  end
  CUDA.device_synchronize(blocking=true)
  assignments
 end
 # Warm the exact dynamic-dispatch, CPU/GPU, stream and synchronization paths.
 reset!();prepare_dispatch!();execute_all!();reset!();prepare_dispatch!();execute_all!()
 for trial=1:5
  reset!();prepare_dispatch!();CUDA.device_synchronize(blocking=true)
  start=time_ns();assignments=execute_all!();elapsed=(time_ns()-start)*1e-9
  out=copy(prefix);ok=true
  for (n,a,x,r,ha,hx,hr,da,dx,dr,ca,cx,gold) in states
   ok&=ha==ca && hx==cx && hr==gold;append!(out,hr)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":1326,\"max_abs_error\":%d,\"max_rel_error\":%d,\"worker_assignments\":[%s],\"timing_region\":\"dynamic_cpu_gpu_task_dispatch_and_device_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",ok ? 0 : 1,ok ? 0 : 1,join(assignments,","),path,length(out))
  ok||error("StreamUM complete task mismatch")
 end
 println("PASS")
end
