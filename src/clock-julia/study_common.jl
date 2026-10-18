using CUDA,Printf
function clock_study(launch!,model)
 words=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==1042
 cases=Any[]
 for fixture in 0:1,b in (1,32,8192)
  a=collect(reinterpret(Float32,words[fixture*512+1:fixture*512+512]));gold=a[1];for x in a;if x<gold;gold=x;end;end
  dout=CUDA.fill(-12345f0,b+16);dtimer=CUDA.fill(Int64(-991827),2b+16)
  push!(cases,(a=CuArray(a),out=dout,timer=dtimer,ov=view(dout,9:b+8),tv=view(dtimer,9:2b+8),blocks=b,gold=reinterpret(UInt32,gold)))
 end
 for c in cases;launch!(c.a,c.ov,c.tv,c.blocks);CUDA.synchronize();end
 for trial in 1:5
  ok=true;elapsed=0.;art=copy(words);raw=Int64[];path_times=Float64[]
  for c in cases
   fill!(c.out,-12345f0);fill!(c.timer,Int64(-991827));CUDA.synchronize();t=time_ns()
   for rep in 1:100;launch!(c.a,c.ov,c.tv,c.blocks);end
   CUDA.synchronize();dt=(time_ns()-t)*1e-9/100;elapsed+=dt;push!(path_times,dt)
   got=reinterpret(UInt32,Array(c.out));timer=Array(c.timer);append!(raw,timer)
   for i in eachindex(got);want=9<=i<=c.blocks+8 ? c.gold : reinterpret(UInt32,-12345f0);ok&=got[i]==want;end
   append!(art,got)
   for i in vcat(1:8,2c.blocks+9:2c.blocks+16);ok&=timer[i]==-991827;push!(art,reinterpret(UInt64,timer[i])%UInt32);end
   for i in 1:c.blocks
    a=timer[8+i];b=timer[8+c.blocks+i];delta=(b%UInt64)-(a%UInt64);format=a>=0&&b>=0;pass=format&&0<delta<0x80000000;ok&=pass;push!(art,UInt32(pass))
   end
  end
  dir=get(ENV,"STUDY_OUTPUT_DIR",pwd());file=joinpath(dir,"output_trial$(trial).bin");write(file,art)
  rawdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","clock-julia-"*model,"raw_timings");mkpath(rawdir);rawfile=joinpath(rawdir,basename(dir)*"_trial$(trial).bin");write(rawfile,raw)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum6_clocked_reduction_hostinclusive_batches_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"raw_timer_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%d},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,elapsed,ok ? "true" : "false",length(art)-length(words),rawfile,length(raw),file,length(art),string(path_times))
 end
end
