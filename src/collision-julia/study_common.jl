using CUDA,Printf
function collision_study(launch!)
 words=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==1282
 cases=Any[]
 for ci in 0:39
  host=fill(Int32(-991827),48);host[9:40]=reinterpret(Int32,words[ci*32+1:ci*32+32]);seen=Set{Int32}();mask=UInt32(0)
  for i in 1:32;if host[8+i] in seen;mask|=UInt32(1)<<(i-1);end;push!(seen,host[8+i]);end
  dinput=CuArray(host);flags=CUDA.fill(Int32(-991827),32);outmask=CUDA.fill(UInt32(0xfedcba98),1)
  push!(cases,(host=host,input=dinput,a=view(dinput,9:40),flags=flags,mask=outmask,goldflag=Int32(length(seen)!=32),goldmask=mask))
 end
 for c in cases;launch!(c.a,c.flags,c.mask);CUDA.synchronize();end
 for trial in 1:5
  art=copy(words);ok=true;elapsed=0.;path_times=Float64[]
  for c in cases
   fill!(c.flags,Int32(-991827));fill!(c.mask,UInt32(0xfedcba98));CUDA.synchronize();t=time_ns()
   for rep in 1:100;launch!(c.a,c.flags,c.mask);end
   CUDA.synchronize();dt=(time_ns()-t)*1e-9/100;elapsed+=dt;push!(path_times,dt)
   flags=Array(c.flags);mask=Array(c.mask)[1];echo=Array(c.input);ok&=all(==(c.goldflag),flags)&&mask==c.goldmask&&echo==c.host
   append!(art,reinterpret(UInt32,flags));push!(art,mask);append!(art,reinterpret(UInt32,echo))
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":3240,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum40_warp32_collision_two_kernel_batches_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,elapsed,ok ? "true" : "false",file,length(art),string(path_times))
 end
end
