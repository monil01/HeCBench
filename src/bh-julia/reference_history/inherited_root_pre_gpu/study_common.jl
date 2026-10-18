using CUDA,Printf
include("study_device.jl")
function bh_study(factory)
 cases=[];prefix=Float32[];expected=reinterpret(Float64,read(joinpath(@__DIR__,"expected_f64.bin")))
 for (N,steps)in [(256,1),(256,3),(1024,1),(1024,3),(2048,1)]
  h=reinterpret(Float32,read(joinpath(@__DIR__,"fixture_$(N).bin")));nn=Int32(max(2*N,24576)-1);nb=Int32(N)
  p=CUDA.zeros(Float32,4*(nn+1));vel=CUDA.zeros(Float32,2*(nn+1));acc=CUDA.zeros(Float32,4*(nn+1))
  start=CUDA.zeros(Int32,nn+1);child=CUDA.zeros(Int32,8*(nn+1));count=CUDA.zeros(Int32,nn+1);sort=CUDA.zeros(Int32,nn+1)
  maxs=CUDA.zeros(Float32,3*48);mins=similar(maxs);radius=CUDA.zeros(Float32,1);bottom=CUDA.zeros(Int32,1);step=similar(bottom);bc=CUDA.zeros(UInt32,1)
  function reset()
   copyto!(p,1,h,1,4*N);copyto!(vel,1,h,4*N+1,2*N);copyto!(acc,1,h,6*N+1,4*N)
  end
  reset();append!(prefix,Array(p)[1:4*N]);append!(prefix,Array(vel)[1:2*N]);append!(prefix,Array(acc)[1:4*N])
  run=factory(nn,nb,steps,start,child,count,sort,p,vel,acc,maxs,mins,radius,bottom,step,bc)
  run();push!(cases,(N,nn,reset,run,p,vel,acc,sort,count))
 end
 allok=true
 for trial in 1:5
  data=copy(prefix);times=Float64[]
  for (n,nn,reset,run,p,vel,acc,sort,count)in cases
   reset();CUDA.device_synchronize(blocking=true);t=time_ns();run();CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9)
   append!(data,Array(p)[1:4*n]);append!(data,Array(vel)[1:2*n]);append!(data,Array(acc)[1:4*n]);sort!(Array(sort)[1:n])==Int32.(0:n-1)||error("permutation");Array(count)[nn+1]==n||error("rootcount")
  end
  out=@view data[length(prefix)+1:end];e=abs.(Float64.(out).-expected);ok=all(isfinite,out)&&all(e.<=1e-5.+1e-4.*abs.(expected));allok&=ok
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"bh_trial$(trial).bin");write(file,data)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"mismatches\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"component_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(expected),count(e.>1e-5.+1e-4.*abs.(expected)),maximum(e),maximum(e./max.(abs.(expected),1e-30)),join(times,","),file,length(data))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oracle")
end
