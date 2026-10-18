using CUDA,Printf
include("study_device.jl")
include("study_topology.jl")
function bh_study(factory,allocate=(T,n)->CUDA.zeros(T,n))
 function guarded(T,n,guards)
  raw=allocate(T,n+64);pattern=reinterpret(T,UInt32(0x5a5a5a5a));fill!(raw,pattern);data=view(raw,33:n+32);fill!(data,zero(T));push!(guards,(raw,n));return data
 end
 cases=[];prefix=Float32[];expected=reinterpret(Float64,read(joinpath(@__DIR__,"expected_f64.bin")))
 for (N,steps)in [(256,1),(256,3),(1024,1),(1024,3),(2048,1)]
  h=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture_$(N).bin"))));nn=Int32(max(2*N,24576)-1);nb=Int32(N)
  guards=[]
  p=guarded(Float32,4*(nn+1),guards);vel=guarded(Float32,2*(nn+1),guards);acc=guarded(Float32,4*(nn+1),guards)
  start=guarded(Int32,nn+1,guards);child=guarded(Int32,8*(nn+1),guards);count=guarded(Int32,nn+1,guards);sort=guarded(Int32,nn+1,guards)
  maxs=guarded(Float32,3*48,guards);mins=guarded(Float32,3*48,guards);radius=guarded(Float32,1,guards);bottom=guarded(Int32,1,guards);step=guarded(Int32,1,guards);bc=guarded(Int32,1,guards)
  function reset()
   copyto!(p,1,h,1,4*N);copyto!(vel,1,h,4*N+1,2*N);copyto!(acc,1,h,6*N+1,4*N)
  end
  reset();append!(prefix,Array(p)[1:4*N]);append!(prefix,Array(vel)[1:2*N]);append!(prefix,Array(acc)[1:4*N])
  run=factory(nn,nb,steps,start,child,count,sort,p,vel,acc,maxs,mins,radius,bottom,step,bc)
  run();push!(cases,(N,nn,reset,run,p,vel,acc,sort,count,start,child,bottom,guards))
 end
 allok=true
 for trial in 1:5
  data=copy(prefix);times=Float64[]
  caseok=true
  for (n,nn,reset,run,p,vel,acc,sort,count,starts,child,bottom,guards)in cases
   reset();CUDA.device_synchronize(blocking=true);t=time_ns();run();CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9)
   append!(data,Array(p)[1:4*n]);append!(data,Array(vel)[1:2*n]);append!(data,Array(acc)[1:4*n]);topok,why,nodes,depth=bh_treecheck(n,nn,Array(p),Array(sort),Array(count),Array(starts),Array(child),Array(bottom)[1]);guardok=true
   for(raw,len)in guards
    left=reinterpret(UInt32,Array(view(raw,1:32)));right=reinterpret(UInt32,Array(view(raw,len+33:len+64)));guardok&=all(==(UInt32(0x5a5a5a5a)),left)&&all(==(UInt32(0x5a5a5a5a)),right)
   end
   println("STUDY_TREE_CHECK trial=",trial," n=",n," topology=",topok," reason=",why," nodes=",nodes," maxdepth=",depth," guards=",guardok," allocations=",length(guards));caseok&=topok&&guardok
  end
  out=@view data[length(prefix)+1:end];e=abs.(Float64.(out).-expected);ok=caseok&&all(isfinite,out)&&all(e.<=1e-5.+1e-4.*abs.(expected));allok&=ok
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"bh_trial$(trial).bin");write(file,data)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"mismatches\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"component_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(expected),Base.count(e.>1e-5.+1e-4.*abs.(expected)),maximum(e),maximum(e./max.(abs.(expected),1e-30)),join(times,","),file,length(data))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oracle")
end
