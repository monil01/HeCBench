const TEST_INDICES=Int32[48427,17148,23627,62548,4431]
const TEST_RANKS=Int32[0,18,346,64917,65463]
function independent_keys()
 state=UInt64(314159265);mult=UInt64(1220703125);mask=(UInt64(1)<<46)-1;keys=Vector{Int32}(undef,65536)
 for i in eachindex(keys)
  total=UInt64(0)
  for j in 1:4;state=(state*mult)&mask;total+=state;end
  keys[i]=Int32((total*UInt64(512))>>46)
 end
 return keys
end
function study(run!)
 original=independent_keys();final=copy(original)
 for iteration in 1:24;final[iteration+1]=iteration;final[iteration+25]=2048-iteration;end
 counts=zeros(Int32,2048);for key in final;counts[key+1]+=1;end;prefix=cumsum(counts);exclusive=prefix.-counts;partial=final[Int.(TEST_INDICES).+1];ordered=sort(final);liveoffset=Int32[sum(counts[1:256i]) for i in 0:7];expected=vcat(original,final,prefix,partial,Int32[120],liveoffset,ordered,exclusive,zeros(Int32,256))
 keys=CUDA.zeros(Int32,65536);hist=CUDA.zeros(Int32,2048);backup=similar(keys);indices=CuArray(TEST_INDICES);ranks=CuArray(TEST_RANKS);part=CUDA.zeros(Int32,5);passed=CUDA.zeros(Int32,1);sums=CUDA.zeros(Int32,256);aux=CUDA.zeros(Int32,256);tmp=CUDA.zeros(Int32,512);verifytmp=CUDA.zeros(Int32,256)
 function trial!()
  args=(keys,hist,backup,indices,ranks,part,passed,sums,aux);CUDA.fill!(passed,0);CUDA.fill!(sums,0);run!(0,args...,Int32(0),tmp,verifytmp);CUDA.synchronize();initial=Array(keys)
  CUDA.synchronize();t=time_ns()
  for iteration in 1:24,stage in 1:7;run!(stage,args...,Int32(iteration),tmp,verifytmp);end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9/24;unsorted=Array(keys);cumulative=Array(hist);partials=Array(part);passes=Array(passed);offsets=Array(sums)[1:8]
  for stage in 8:10;run!(stage,args...,Int32(24),tmp,verifytmp);end
  CUDA.synchronize();got=vcat(initial,unsorted,cumulative,partials,passes,offsets,Array(keys),Array(hist),Array(aux));ok=got==expected
  return got,ok,elapsed
 end
 trial!()
 for trial in 1:5
  got,ok,elapsed=trial!();path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":200974,\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"native24iterations_seven_gpu_rank_kernels_mean\",\"jit_excluded\":true,\"internal_repeats\":24,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":200974,\"byte_order\":\"little\"}}")
  ok||error("NASfullrng/rank/prefix/scatter/verificationfailed")
 end
end
