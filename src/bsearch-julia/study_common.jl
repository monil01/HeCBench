using CUDA,Printf
const BS_SIZES=(2,3,8,9,17,32,33,257,1025)
const BS_GUARD=UInt64(0xD15EA5E5A55A77CC)
function study_main(launch!)
 words=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==11988
 arrays=Any[];queries=Any[];ns=Int64[];expected=fill(BS_GUARD,528,72);offset=0;case=0
 for T in (Float32,Float64),sz in BS_SIZES
  case+=1;part=words[offset+1:offset+sz+512];decoded=T===Float32 ? collect(reinterpret(Float32,UInt32.(part))) : collect(reinterpret(Float64,part));a=decoded[1:sz];z=decoded[sz+1:end];n=Int64(sz-1)
  @assert all(isfinite,a)&&issorted(a)&&all(diff(a).>0)&&all(a[1].<=z.<a[end])
  da=CuArray(a);dz=CuArray(z);@assert Array(da)==a&&Array(dz)==z;push!(arrays,da);push!(queries,dz);push!(ns,n)
  for mode in 1:4,i in 1:512
   idx=searchsortedlast(a,z[i])-1;@assert 0<=idx<n&&a[idx+1]<=z[i]<a[idx+2];expected[8+i,(case-1)*4+mode]=UInt64(idx)
  end
  offset+=sz+512
 end
 @assert offset==length(words)
 out=CUDA.fill(BS_GUARD,528,72);views=[view(out,9:520,j) for j in 1:72]
 for case in 1:18,mode in 1:4;launch!(arrays[case],queries[case],views[(case-1)*4+mode],ns[case],Val(mode));end
 CUDA.synchronize()
 for trial in 1:5
  fill!(out,BS_GUARD);CUDA.synchronize();t=time_ns()
  for rep in 1:5,case in 1:18,mode in 1:4;launch!(arrays[case],queries[case],views[(case-1)*4+mode],ns[case],Val(mode));end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9/5;got=Array(out);ok=got==expected;bad=count(got.!=expected);art=vcat(words,vec(got));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":38016,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"complete72path_binarysearch_suite_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":5,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":50004,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",bad,file)
 end
end
