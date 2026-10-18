include(joinpath(@__DIR__,"study_fixture.jl"))
function study(allocate,launch)
 cases=[];prefix=Float64[]
 for c in 0:1
  a,b=fixture(c);da=allocate(Float32,length(a));db=allocate(Float32,length(b));copyto!(da,a);copyto!(db,b);o=allocate(Float32,128);idx=allocate(Int32,128);append!(prefix,Float64.(a));append!(prefix,Float64.(b));append!(prefix,Float64[128,128,3]);push!(cases,(a,b,da,db,o,idx))
 end
 function run(c,k)
  a,b,da,db,o,idx=c;fill!(o,0);fill!(idx,-1);CUDA.synchronize(CUDA.context());start=time_ns();for rep in 1:3;launch(k,da,db,o,idx);end;CUDA.synchronize(CUDA.context());t=(time_ns()-start)*1e-9/3;h=Array(o);i=Array(idx);s,j=oracle(a,b,k);return vcat(Float64.(h),Float64.(i)),t,h==s&&i==j
 end
 for c in cases,k in 1:10;run(c,k);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;components=Float64[]
  for c in cases,k in 1:10;v,t,pass=run(c,k);append!(out,v);elapsed+=t;push!(components,t);ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(components,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum20_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("MatchFullScoreIndexOracle")
 end
end
