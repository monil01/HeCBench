using CUDA,Printf
include(joinpath(@__DIR__,"study_fixture.jl"))
const ST_CAP=UInt32(64*1024*1024)
const ST_EMPTY=typemax(UInt32)
function gold_snapshots()
 dict=Dict{UInt32,UInt32}();result=Vector{Vector{Tuple{UInt32,UInt32}}}()
 for stage in 1:5
  keys,values=STUDY_BATCHES[stage]
  for (k,v) in zip(keys,values)
   if stage==4;Base.delete!(dict,k);else;dict[k]=v;end
  end
  if stage>=3;push!(result,sort!(collect((k,v) for (k,v) in dict)));end
 end
 result
end
function study_main(insert!,delete!,iterate!,label)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);gold=gold_snapshots();tk=CUDA.fill(ST_EMPTY,Int(ST_CAP));tv=CUDA.fill(ST_EMPTY,Int(ST_CAP));batchdev=[(CuArray(a),CuArray(v)) for (a,v) in STUDY_BATCHES];outputs=[(CUDA.fill(ST_EMPTY,1027),CUDA.fill(ST_EMPTY,1027),CUDA.zeros(UInt32,1)) for _ in 1:3]
 prefix=UInt32[ST_CAP,ST_EMPTY,5,3,1027]
 for (keys,values) in STUDY_BATCHES;push!(prefix,UInt32(length(keys)));append!(prefix,keys);append!(prefix,values);end
 function path!(which)
  if which<=3;d=batchdev[which];insert!(tk,tv,d[1],d[2],UInt32(length(d[1])))
  elseif which==4;d=outputs[1];iterate!(tk,tv,d...)
  elseif which==5;d=batchdev[4];delete!(tk,tv,d[1],UInt32(length(d[1])))
  elseif which==6;d=outputs[2];iterate!(tk,tv,d...)
  elseif which==7;d=batchdev[5];insert!(tk,tv,d[1],d[2],UInt32(length(d[1])))
  else;d=outputs[3];iterate!(tk,tv,d...);end
 end
 function reset!()
  CUDA.fill!(tk,ST_EMPTY);CUDA.fill!(tv,ST_EMPTY)
  for d in outputs;CUDA.fill!(d[1],ST_EMPTY);CUDA.fill!(d[2],ST_EMPTY);CUDA.fill!(d[3],UInt32(0));end
  CUDA.synchronize()
 end
 reset!();for which in 1:8;path!(which);end;CUDA.synchronize()
 rawdir=abspath(joinpath(@__DIR__,"..","..",".porting-state","portable_julia","raw_sidecars","linearprobing",label*"_"*string(getpid())*"_"*string(time_ns())));mkpath(rawdir)
 for trial in 1:5
  times=zeros(Float64,8);gct=0.0;gcp=0
  for _ in 1:rep
   reset!()
   for which in 1:8
    CUDA.synchronize();gb=Base.gc_num();st=time_ns();path!(which);CUDA.synchronize();times[which]+=(time_ns()-st)*1e-9/rep;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   end
  end
  canonical=UInt32[];raw=UInt32[];ok=true
  for (d,expected) in zip(outputs,gold)
   count=Int(Array(d[3])[1]);ok&=0<=count<=1027;0<=count<=1027||error("outputbounds")
   keys=Array(d[1])[1:count];values=Array(d[2])[1:count];pairs=collect(zip(keys,values));push!(raw,UInt32(count));for (k,v) in pairs;push!(raw,k,v);end
   sort!(pairs);ok&=pairs==expected&&length(unique(first.(pairs)))==count;push!(canonical,UInt32(count));for (k,v) in pairs;push!(canonical,k,v);end
  end
  rawfile=joinpath(rawdir,"raw_trial$(trial).bin");write(rawfile,raw);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");art=vcat(prefix,canonical);write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum8ordered_insert_delete_iterate_kernels_resetexcluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"raw_unordered_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(canonical),rep,join(times,","),gct,gcp,rawfile,length(raw),file,length(art));ok||error("fullthreeunordered_numericdictionary_snapshots")
 end
end
