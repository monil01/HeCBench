using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
const ST_SOURCE=collect(reinterpret(Float32,read(joinpath(@__DIR__,"source_cpu_outputs.bin"))))
const ST_NB=Int(ST_INPUT[1]);const ST_N=Int(ST_INPUT[2]);const ST_NP=ST_N+17
const ST_CASES=Tuple((UInt32(ST_INPUT[j])|(UInt32(ST_INPUT[j+1])<<16),Int(ST_INPUT[j+2])) for j in (4,7,10))
const ST_POOL=ST_INPUT[13:12+ST_NP];const ST_CHI=ST_INPUT[13+ST_NP:end]
const ST_ENERGY=[sum(abs2,Float64.(ST_POOL[(b-1)*2048+1:b*2048])) for b in 1:ST_NB]
function st_reset!(d)
 copyto!(d[1],ST_POOL);CUDA.fill!(d[2],-12345f0);CUDA.synchronize()
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);d=(CuArray(ST_POOL),CUDA.fill(-12345f0,ST_NP),CuArray(ST_CHI))
 for (seed,steps) in ST_CASES
  st_reset!(d);for _ in 1:steps;launch!(d,seed);end;CUDA.synchronize()
 end
 for trial in 1:5
  times=Float64[];got=Float32[];normdrift=Float64[];gct=0.;gcp=0
  for (seed,steps) in ST_CASES
   elapsed=0.
   for _ in 1:rep
    st_reset!(d);gb=Base.gc_num();t=time_ns()
    for _ in 1:steps;launch!(d,seed);end
    CUDA.synchronize();elapsed+=(time_ns()-t)*1e-9;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   end
   pool=Array(d[1]);append!(got,pool);append!(got,Array(d[2]));push!(times,elapsed/rep)
   push!(normdrift,maximum(abs(sum(abs2,Float64.(pool[(b-1)*2048+1:b*2048]))-ST_ENERGY[b])/ST_ENERGY[b] for b in 1:ST_NB))
  end
  ok=true;ma=0.;mr=0.;sourceerr=0.
  for i in eachindex(got)
   e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);mr=max(mr,e/max(abs(ST_GOLD[i]),1e-12));sourceerr=max(sourceerr,abs(Float64(got[i])-Float64(ST_SOURCE[i])));ok&=isfinite(got[i])&&(ST_GOLD[i]==-12345 ? got[i]==-12345 : e<=1e-3)&&abs(got[i]-ST_SOURCE[i])<=1f-3
  end
  ok&=maximum(normdrift)<=1e-5
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"source_cpu_max_abs_error\":%.17g,\"max_relative_pool_squared_norm_drift\":%.17g,\"timing_region\":\"sum3_stateful_chains_1_7_31_kernel_calls_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,sourceerr,maximum(normdrift),rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
