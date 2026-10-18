const SENT=UInt8(0xa5)
function study(allocate,launch)
 @assert sizeof(Properties)==40
 fixture=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixtures_u32.bin"))));lo=read(joinpath(@__DIR__,"oracle_lower_u8.bin"));hi=read(joinpath(@__DIR__,"oracle_upper_u8.bin"));states=[];off=0;guard=only(reinterpret(Properties,fill(UInt32(0xdeadbeef),10)))
 while off<length(fixture)
  w,h,mode,reps=Int.(fixture[off+1:off+4]);n=w*h;prop=only(reinterpret(Properties,fixture[off+5:off+14]));raw=vcat(SENT,UInt8.(fixture[off+15:off+14+3n]),SENT);ph=[guard,prop,guard];a=allocate(UInt8,3n+2);b=allocate(UInt8,3n+2);devprop=allocate(Properties,3);copyto!(a,raw);copyto!(devprop,ph);push!(states,(w,h,mode,reps,a,b,devprop,raw,ph));off+=14+3n
 end
 function run(c)
  w,h,mode,reps,a,b,devprop,raw,ph=c;n=w*h;fill!(b,SENT);src=reshape(view(a,2:3n+1),3,n);dst=reshape(view(b,2:3n+1),3,n);pr=view(devprop,2:2);CUDA.synchronize(CUDA.context());start=time_ns()
  for rep in 1:reps;launch(src,dst,pr,w,h);end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/reps;got=Array(b);back=Array(a);pb=Array(devprop);ok=back==raw&&reinterpret(UInt32,pb)==reinterpret(UInt32,ph)&&got[1]==SENT&&got[end]==SENT;guards=vcat(UInt32[back[1],back[end],got[1],got[end]],collect(reinterpret(UInt32,pb[[1,3]])));return got[2:end-1],guards,elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true
  for c in states
   vals,guards,t,pass=run(c);ok&=pass&&all(lo[oi+j]<=vals[j]<=hi[oi+j] for j in eachindex(vals));oi+=length(vals);append!(out,UInt32.(vals));append!(out,guards);elapsed+=t
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum4_source_barrel_5repeat_means_globalproperties_complete\",\"jit_excluded\":true,\"internal_repeats\":5,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("QuantizedBarrelPerChannelDerivedIntervalIdentityBlackBoundaryGuards")
 end
end
