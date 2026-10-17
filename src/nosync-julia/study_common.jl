using CUDA,Libdl
function study(allocate)
 lib=Libdl.dlopen(joinpath(pwd(),"libnosync_study.so"));create=Libdl.dlsym(lib,:study_create_stream);destroy=Libdl.dlsym(lib,:study_destroy_stream);pipeline=Libdl.dlsym(lib,:study_nosync);version=Libdl.dlsym(lib,:study_thrust_version);println("PinnedThrustVersion ",ccall(version,Cint,()));stream=Ref{Ptr{Cvoid}}(C_NULL);@assert ccall(create,Cint,(Ref{Ptr{Cvoid}},),stream)==0
 cases=[];prefix=Int32[]
 for n in (1025,65535)
  vec=allocate(Int32,n);res=allocate(Int32,n);zeroshost=zeros(Int32,n);append!(prefix,zeroshost);append!(prefix,zeroshost);append!(prefix,Int32[n,3,0,1]);sum=Int32(n*(n-1)÷2);want=vcat(Int32.(0:n-1),Int32[i*(i+1)÷2 for i in 0:n-1],Int32[sum,sum,0]);push!(cases,(n,vec,res,zeroshost,want,zeros(Int32,3)))
 end
 function case!(c)
  n,vec,res,zero,want,result=c;copyto!(vec,zero);copyto!(res,zero);CUDA.synchronize(CUDA.context());t=time_ns()
  for _ in 1:3
   status=ccall(pipeline,Cint,(CuPtr{Int32},CuPtr{Int32},Cint,Ptr{Cvoid},Ptr{Int32}),vec,res,Int32(n),stream[],result);status==0||error("ThrustInteropStatus",status)
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9/3;got=vcat(Array(vec),Array(res),result);return got,elapsed,got==want
 end
 for c in cases;got,t,ok=case!(c);ok||error("WarmupMismatch");end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true
  for c in cases;got,t,pass=case!(c);append!(out,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum_two_preallocated_three_thrust_pipeline_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullSequenceScanReductionMismatch")
 end
 @assert ccall(destroy,Cint,(Ptr{Cvoid},),stream[])==0
end
