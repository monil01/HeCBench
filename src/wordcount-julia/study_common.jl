function fixtures()
 [UInt8[],UInt8[65],UInt8[32],UInt8[v for x in 91:96 for v in (x,32)],UInt8.(0:255),UInt8[mod(i*73+div(i,17),256) for i in 0:65572],UInt8[[0,65,91,32,122,96,127,255,97,0,90][mod(i,11)+1] for i in 0:1024]]
end
alpha(x)=65<=x<=122
function count_reference(x)
 sum(alpha(x[i])&&(i==1||!alpha(x[i-1])) for i in eachindex(x);init=0)
end
function study(allocate,countfn)
 cases=[];prefix=UInt32[]
 for x in fixtures()
  d=allocate(x);actual=Array(d);actual==x||error("InputCopyMismatch");append!(prefix,UInt32.(actual));append!(prefix,UInt32[length(x),3]);push!(cases,(d,length(x),count_reference(x)))
 end
 for(d,n,ref)in cases;countfn(d,n);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;components=Float64[]
  for(d,n,ref)in cases
   CUDA.synchronize(CUDA.context());start=time_ns();v=0
   for rep in 1:3;v=countfn(d,n);end
   CUDA.synchronize(CUDA.context());t=(time_ns()-start)*1e-9/3;elapsed+=t;push!(components,t);push!(out,UInt32(v));ok&=v==ref
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(components,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum7_three_preallocated_count_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("CountScalarOracle")
 end
end
