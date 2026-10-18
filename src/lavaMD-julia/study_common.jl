const FSENT=-8192.25f0
const MSENT=UInt64(0xdeadbeef12345678)
function study(allocate,launch)
 f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));budget=collect(reinterpret(Float64,read(joinpath(@__DIR__,"error_budget_f64.bin"))));states=[];pos=1
 @assert sizeof(StudyFour)==16
 while pos<=length(f)
  B,mode=Int.(f[pos:pos+1]);alpha=Float32(f[pos+2]);np,mw=Int.(f[pos+3:pos+4]);pos+=5
  words=UInt32.(f[pos:pos+mw-1]);pos+=mw;mh=vcat(fill(MSENT,4),collect(reinterpret(UInt64,words)),fill(MSENT,4));rh=vcat(fill(FSENT,4),Float32.(f[pos:pos+4np-1]),fill(FSENT,4));pos+=4np;qh=vcat(fill(FSENT,4),Float32.(f[pos:pos+np-1]),fill(FSENT,4));pos+=np
  ma=allocate(UInt64,length(mh));ra=allocate(Float32,length(rh));qa=allocate(Float32,length(qh));fa=allocate(Float32,4np+8);copyto!(ma,mh);copyto!(ra,rh);copyto!(qa,qh);fill!(fa,FSENT);mv=view(ma,5:length(ma)-4);rv=reinterpret(StudyFour,view(ra,5:4np+4));qv=view(qa,5:np+4);fv=reinterpret(StudyFour,view(fa,5:4np+4));rawfv=view(fa,5:4np+4)
  push!(states,(B,mode,alpha,np,mh,rh,qh,ma,ra,qa,fa,mv,rv,qv,fv,rawfv))
 end
 for (B,mode,alpha,np,mh,rh,qh,ma,ra,qa,fa,mv,rv,qv,fv,rawfv) in states
  fill!(rawfv,0f0);launch(mv,rv,qv,fv,alpha,B^3);CUDA.synchronize(CUDA.context())
 end
 for trial in 1:5
  payload=copy(f);ok=true;elapsed=0.;times=Float64[];oi=1;maxerr=0.;maxratio=0.;imbalances=Float64[]
  for (B,mode,alpha,np,mh,rh,qh,ma,ra,qa,fa,mv,rv,qv,fv,rawfv) in states
   fill!(fa,FSENT);fill!(rawfv,0f0);CUDA.synchronize(CUDA.context());start=time_ns();launch(mv,rv,qv,fv,alpha,B^3);CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9;elapsed+=tm;push!(times,tm)
   mb=Array(ma);rb=Array(ra);qb=Array(qa);fb=Array(fa);ok&=mb==mh&&reinterpret(UInt32,rb)==reinterpret(UInt32,rh)&&reinterpret(UInt32,qb)==reinterpret(UInt32,qh)&&all(x==FSENT for x in fb[1:4])&&all(x==FSENT for x in fb[end-3:end]);append!(payload,Float64.(reinterpret(UInt32,mb)));append!(payload,Float64.(rb));append!(payload,Float64.(qb));append!(payload,Float64.(fb))
   got=Float64.(fb[5:end-4]);ref=oracle[oi:oi+4np-1];eb=budget[oi:oi+4np-1];oi+=4np;errors=abs.(got.-ref);maxerr=max(maxerr,maximum(errors));maxratio=max(maxratio,maximum(errors./eb));ok&=all(isfinite,got)&&all(errors.<=eb)
   if mode==0
    total=sum(Float64.(qh[5:end-4]));ok&=all(got[4i+1]==total&&got[4i+2]==0&&got[4i+3]==0&&got[4i+4]==0 for i in 0:np-1)
   end
   for axis in 2:4
    imbalance=abs(sum(Float64(qh[i+4])*got[4(i-1)+axis] for i in 1:np));allow=sum(abs(Float64(qh[i+4]))*eb[4(i-1)+axis] for i in 1:np)+1e-9;ok&=imbalance<=allow;push!(imbalances,imbalance)
   end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",maxerr,",\"max_forward_error_budget_ratio\":",maxratio,",\"weighted_force_imbalances\":",imbalances,",\"timing_region\":\"sum3_lavaMD_original_single_kernel_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":3,\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullLavaMDFloat64OracleForwardModelReadonlyGuardsExactControl")
 end
end
