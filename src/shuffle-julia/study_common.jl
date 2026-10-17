using CUDA,Printf
const STUDY_N=8192
const STUDY_SGS=(8,16,32)
const STUDY_PATTERNS=(UInt32(0xdeadbeef),UInt32(123456789))
function study_inputs()
 a=Float32[Float32(i)*10f0 for i in 0:STUDY_N-1]
 b=Float32[i%257==0 ? -0f0 : Float32((i*13)%16381-8190)/16f0 for i in 0:STUDY_N-1]
 (a,b)
end
function study_paths()
 paths=Tuple{Int,Int,UInt32,Int}[(1,s,UInt32(0),0) for s in STUDY_SGS]
 for pattern in STUDY_PATTERNS,s in STUDY_SGS;push!(paths,(2,s,pattern,0));end
 for input in 1:2,s in STUDY_SGS;push!(paths,(3,s,UInt32(0),input));end
 paths
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);inputs=study_inputs();din=map(CuArray,inputs);paths=study_paths();outs=[z[1]<=2 ? CuArray{Int32}(undef,256) : CuArray{Float32}(undef,STUDY_N) for z in paths];prefix=vcat(Float64[STUDY_N,256,reinterpret(Int32,STUDY_PATTERNS[1]),reinterpret(Int32,STUDY_PATTERNS[2]),8,16,32],Float64.(inputs[1]),Float64.(inputs[2]))
 for (z,out) in zip(paths,outs);launch!(z[1],z[2],z[3],out,din[max(z[4],1)]);end;CUDA.synchronize()
 for trial in 1:5
  times=Float64[];gct=0.0;gcp=0
  for (z,out) in zip(paths,outs)
   CUDA.synchronize();gb=Base.gc_num();st=time_ns()
   for _ in 1:rep;launch!(z[1],z[2],z[3],out,din[max(z[4],1)]);end
   CUDA.synchronize();push!(times,(time_ns()-st)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
  end
  got=Float64[];ok=true
  for (z,out) in zip(paths,outs)
   a=Array(out);kind,s,pattern,input=z
   if kind==1
    ok&=all(==(Int32(s*(s-1)÷2)),a)
   elseif kind==2
    ok&=all(==(reinterpret(Int32,pattern)),a)
   else
    # Independent contiguousgroup reversal, exact Float32bits includingnegativezero.
    gold=similar(inputs[input])
    for base in 0:s:STUDY_N-1,j in 0:s-1;gold[base+j+1]=inputs[input][base+s-j];end
    ok&=reinterpret(UInt32,a)==reinterpret(UInt32,gold)
   end
   append!(got,Float64.(a))
  end
  art=vcat(prefix,got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum15paths_XOR_broadcast_blockreverse_kernelonly\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),rep,join(times,","),gct,gcp,file,length(art));ok||error("allsubgroupXORsums/broadcastpatterns/reversalFloat32bits")
 end
end
