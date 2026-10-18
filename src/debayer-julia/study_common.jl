const SENT=UInt8(0xa5)
function study(allocate,launch)
 fixture=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixtures_u32.bin"))));oracle=read(joinpath(@__DIR__,"oracle_u8.bin"));states=[];off=0
 while off<length(fixture)
  w,h,pat,mode,reps=Int.(fixture[off+1:off+5]);n=w*h;raw=vcat(SENT,UInt8.(fixture[off+6:off+5+n]),SENT);a=allocate(UInt8,n+2);b=allocate(UInt8,4n+2);copyto!(a,raw);push!(states,(w,h,pat,mode,reps,a,b,raw));off+=5+n
 end
 function run(c)
  w,h,pat,mode,reps,a,b,raw=c;n=w*h;fill!(b,SENT);CUDA.synchronize(CUDA.context());start=time_ns()
  for rep in 1:reps
   CUDA.memset(pointer(b,2),UInt8(0),4n)
   launch(view(a,2:n+1),view(b,2:4n+1),Int32(w),Int32(h),Int32(pat))
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/reps;got=Array(b);back=Array(a);ok=back==raw&&got[1]==SENT&&got[end]==SENT;return got[2:end-1],UInt32[back[1],back[end],got[1],got[end]],elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true
  for c in states
   vals,guards,t,pass=run(c);ok&=pass&&vals==oracle[oi+1:oi+length(vals)];oi+=length(vals);append!(out,UInt32.(vals));append!(out,guards);elapsed+=t
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum5_source_memset_plus_demosaic_5repeat_means_complete\",\"jit_excluded\":true,\"internal_repeats\":5,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("ExactIntegerMHCAllRGBPaddingInputAndGuards")
 end
end
