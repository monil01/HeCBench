using CUDA,Printf
const FIXTURE=reinterpret(Int32,read(joinpath(@__DIR__,"fixture_csr.bin")))
const N=Int(FIXTURE[1]);const E=Int(FIXTURE[2]);const IDX=collect(FIXTURE[3:3+N]);const LIST=collect(FIXTURE[4+N:end])
function properties(raw)
 p=Int32[(0<=x<N) ? 1 : 0 for x in raw]
 for v in 0:N-1,k in Int(IDX[v+1])+1:Int(IDX[v+2]);push!(p,raw[v+1]!=raw[Int(LIST[k])+1] ? 1 : 0);end
 p
end
function study_main(model,prepare,execute,color_array)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);state=prepare();execute(state);CUDA.synchronize()
 side=joinpath("/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/gc",model*"_"*string(getpid())*"_"*string(time_ns()));mkpath(side)
 for trial in 1:5
  iterations=Int[];CUDA.synchronize();gb=Base.gc_num();st=time_ns()
  for _ in 1:rep;push!(iterations,execute(state));end
  CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep;ga=Base.gc_num();raw=Array(color_array(state));props=properties(raw);ok=all(==(1),props)&&length(unique(raw[1:40]))==40;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");artifact=vcat(collect(FIXTURE),props);write(file,artifact);sidefile=joinpath(side,"raw_trial$(trial).bin");write(sidefile,raw)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"timing_region\":\"complete_graph_coloring_including_resets_and_required_convergence_hostchecks\",\"jit_excluded\":true,\"internal_repeats\":%d,\"number_distinct_colors\":%d,\"max_color_plus_one\":%d,\"host_convergence_rounds\":[%s],\"native_device_loop_iterations\":null,\"raw_color_sidecar\":\"%s\",\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(props),count(!=(1),props),rep,length(unique(raw)),maximum(raw)+1,join(iterations,","),sidefile,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,length(artifact))
 end
end
