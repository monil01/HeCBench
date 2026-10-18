# Source/input-derived discrete FMA auxiliary rule; see identity_FMA_secondary_contract.json.
function identity_fma_allowed(value,idx0,kid,hh,sh)
 if 4<=kid<=23
  pkg,within=divrem(idx0,392);e,within=divrem(within,8);component,lane=divrem(within,4)
  opposite=pkg*392+8*e+(1-component)*4+lane
 else
  opposite=xor(idx0,1)
 end
 x=Float64(sh[opposite+5]);d=Float64(hh[5]);product=d*x
 residual=Float32(Float64(Float32(product))-product)
 return iszero(value)||abs(value)==abs(residual)
end
const SENT=-8192.25f0
function study(allocate,launch)
 f=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];i=1
 @assert sizeof(StudyV4)==16&&sizeof(StudyV2)==8
 while i<=length(f)
  n,mode,kid,dim,reps=Int.(f[i:i+4]);@assert f[i+5]==HDT;i+=6;hlen=98;slen=98n;hh=vcat(fill(SENT,4),f[i:i+hlen-1],fill(SENT,4));i+=hlen;sh=vcat(fill(SENT,4),f[i:i+slen-1],fill(SENT,4));i+=slen;ha=allocate(Float32,hlen+8);sa=allocate(Float32,slen+8);oa=allocate(Float32,slen+8);copyto!(ha,hh);copyto!(sa,sh);fill!(oa,SENT);hv=view(ha,5:hlen+4);sv=view(sa,5:slen+4);ov=view(oa,5:slen+4)
  if kid in 16:23;sv=reinterpret(StudyV4,sv);ov=reinterpret(StudyV4,ov);end
  if kid==24;hv=reinterpret(StudyV2,hv);sv=reinterpret(StudyV2,sv);ov=reinterpret(StudyV2,ov);end
  push!(states,(n,mode,kid,reps,hh,sh,ha,sa,oa,hv,sv,ov,zeros(Float32,slen),view(oa,5:slen+4)))
 end
 for c in states
  n,mode,kid,reps,hh,sh,ha,sa,oa,hv,sv,ov,zero,rawout=c;copyto!(rawout,zero);launch(kid,sv,ov,hv,n);CUDA.synchronize(CUDA.context())
 end
 for trial in 1:5
  payload=copy(f);elapsed=0.;ok=true;oi=1;err=0.;times=Float64[]
  for c in states
   n,mode,kid,reps,hh,sh,ha,sa,oa,hv,sv,ov,zero,rawout=c;fill!(oa,SENT);tm=0.
   for rep in 1:reps
    copyto!(rawout,zero);CUDA.synchronize(CUDA.context());start=time_ns();launch(kid,sv,ov,hv,n);CUDA.synchronize(CUDA.context());tm+=(time_ns()-start)*1e-9
   end
   t=tm/reps;elapsed+=t;push!(times,t);hback=Array(ha);sback=Array(sa);out=Array(oa);ok&=reinterpret(UInt32,hback)==reinterpret(UInt32,hh)&&reinterpret(UInt32,sback)==reinterpret(UInt32,sh)&&all(x==SENT for x in out[1:4])&&all(x==SENT for x in out[end-3:end]);append!(payload,hback);append!(payload,sback)
   if kid>0
    got=Float64.(out[5:end-4]);expected=oracle[oi:oi+length(got)-1];oi+=length(got);errors=abs.(got.-expected);err=max(err,maximum(errors));ok&=all(errors.<=1e-6.+1e-4.*abs.(expected));if mode==2;ok&=all(identity_fma_allowed(out[j+4],j-1,kid,hh,sh) for j in eachindex(got));end;append!(payload,out)
   else
    # Unchanged emptykernel: no fabricated numericoutputs, only outputcanaries pluslaunchcompletiontime.
    append!(payload,out[1:4]);append!(payload,out[end-3:end])
   end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",err,",\"timing_region\":\"sum75_original_1001_repeat_kernel_means_complete\",\"jit_excluded\":true,\"internal_repeats\":1001,\"timed_kernel_calls\":75075,\"numeric_cases\":72,\"execution_only_cases\":3,\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("AllActualCommutatorVariantsIndependentOracleReadonlyCanaries")
 end
end
