function execute!(run!,host,f)
 n=length(host);per=n÷f
 @sync for lane in 1:f
  Threads.@spawn begin
   start=(lane-1)*per+1;stop=lane*per
   initial=Int32.(start-1:stop-1);device=CuArray(initial)
   run!(device,Int32(3),Int32(37));CUDA.synchronize()
   copyto!(view(host,start:stop),Array(device));CUDA.unsafe_free!(device)
  end
 end
 return
end
function study(run!)
 Threads.nthreads()>=32||error("Requires32Juliahostthreads")
 n=262144;host=zeros(Int32,n);expected=Int32[i-1+sum(mod(j,3) for j in 0:36) for i in 1:n]
 for f in (1,2,4,8,16,32);execute!(run!,host,f);end
 for trial in 1:5
  total=0.;got=Int32[]
  for outer in 1:2,f in (1,2,4,8,16,32)
   CUDA.synchronize();t=time_ns();execute!(run!,host,f);CUDA.synchronize();total+=(time_ns()-t)*1e-9;append!(got,host)
  end
  expectedall=repeat(expected,12);ok=got==expectedall;err=maximum(abs.(got.-expectedall));path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"sum_twelve_host_concurrency_end_to_end_cases\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Host-concurrency array oracle failed")
 end
end
