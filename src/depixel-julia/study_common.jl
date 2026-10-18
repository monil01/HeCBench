using CUDA,Printf
function depixel_study(launch!)
 words=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(words)==9998&&length(gold)==17200
 cases=Any[];offset=0
 for ci in 0:6
  w=Int32(words[9985+2ci]);h=Int32(words[9986+2ci]);n=Int(w*h);rgb=vcat(fill(-12345f0,24),reinterpret(Float32,words[offset+1:offset+3n]),fill(-12345f0,24));offset+=3n
  da=CuArray(rgb);tmp=CUDA.fill(UInt32(0xfedcba98),n+16);out=CUDA.fill(UInt32(0xfedcba98),n+16)
  push!(cases,(w=w,h=h,n=n,rgb=da,a=view(da,25:3n+24),tmp=tmp,t=view(tmp,9:n+8),out=out,o=view(out,9:n+8)))
 end
 @assert offset==9984
 for c in cases;launch!(c.a,c.t,c.o,c.w,c.h,c.n);CUDA.synchronize();end
 for trial in 1:5
  elapsed=0.;art=copy(words);path_times=Float64[]
  for c in cases
   fill!(c.tmp,UInt32(0xfedcba98));fill!(c.out,UInt32(0xfedcba98));CUDA.synchronize();t=time_ns()
   for rep in 1:100;launch!(c.a,c.t,c.o,c.w,c.h,c.n);end
   CUDA.synchronize();dt=(time_ns()-t)*1e-9/100;elapsed+=dt;push!(path_times,dt)
   append!(art,Array(c.tmp));append!(art,Array(c.out));append!(art,reinterpret(UInt32,Array(c.rgb)))
  end
  ok=art[9999:end]==gold;@assert length(art)==27198
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":17200,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum7_connect_crossgraph_two_kernel_batches_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":27198,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,elapsed,ok ? "true" : "false",file,string(path_times))
 end
end
