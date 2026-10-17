const STUDY_CASES=((4096,false),(65540,true))
const STUDY_REPEATS=3
function fixture(n,full)
 [full ? (i%17==0 ? typemax(UInt32) : i%19==0 ? UInt32(0) : (UInt32(i)*UInt32(2654435761))⊻(UInt32(i>>3)*UInt32(2246822519))) : UInt32(i%16) for i in 0:n-1]
end
function expected(input)
 sorted=sort(input);scratch=sort(sorted;by=x->x&UInt32(0x0fffffff));n=length(input);region=(n÷4)÷64*4;counts=zeros(UInt32,16,64)
 for group in 0:63
  start=group*region;stop=group==63 ? n : start+region
  for i in start:stop-1;counts[Int(scratch[i+1]>>28)+1,group+1]+=1;end
 end
 offsets=UInt32[];seed=UInt32(0)
 for digit in 1:16,group in 1:64;push!(offsets,seed);seed+=counts[digit,group];end
 vcat(sorted,scratch,offsets)
end
function study(run!)
 cases=[];prefix=UInt32[]
 for(n,full)in STUDY_CASES
  input=fixture(n,full);append!(prefix,input);append!(prefix,UInt32[n,64,STUDY_REPEATS]);push!(cases,(input,CuArray(input),CUDA.zeros(UInt32,n),CUDA.zeros(UInt32,1024),expected(input)))
 end
 function case!(args;repeats=STUDY_REPEATS)
  input,a,b,sums,want=args;copyto!(a,input);CUDA.fill!(b,0);CUDA.fill!(sums,0);CUDA.synchronize();t=time_ns()
  for _ in 1:repeats;run!(a,b,sums,Int32(length(input)));end
  CUDA.synchronize();elapsed=(time_ns()-t)*1e-9/repeats;got=vcat(Array(a),Array(b),Array(sums));return got,elapsed,got==want
 end
 # Warmup uses full3pass ordering because scratchstabletieorderisdefinedbyalready-sorteddatainlastpass.
 for args in cases;case!(args);end
 for trial in 1:5
  values=copy(prefix);elapsed=0.;ok=true
  for args in cases;got,t,pass=case!(args);append!(values,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":210962,\"max_abs_error\":0,\"timing_region\":\"sum_two_cases_three_full_radix_pass_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":210962,\"byte_order\":\"little\"}}")
  ok||Base.error("RadixFullSortScratchScanOracleFailed")
 end
end
