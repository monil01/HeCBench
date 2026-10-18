const SENT=-8192.25
function study(allocate,launch)
 f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];pos=1
 while pos<=length(f)
  n,mode,reps=Int.(f[pos:pos+2]);dx=Tuple(f[pos+3:pos+5]);pos+=6;v=n^3;inputs=Vector{Float64}[];raw=[]
  for j in 1:8
   h=vcat(fill(SENT,4),f[pos:pos+v-1],fill(SENT,4));pos+=v;d=allocate(Float64,v+8);copyto!(d,h);push!(inputs,h);push!(raw,d)
  end
  outs=[allocate(Float64,v+8) for _ in 1:5];foreach(a->fill!(a,SENT),outs);ins=Tuple(view(a,5:v+4) for a in raw);flux=Tuple(view(a,5:v+4) for a in outs);zero=zeros(Float64,v)
  push!(states,(n,mode,reps,dx,v,inputs,raw,outs,ins[1:4],ins[5:8],flux,zero))
 end
 for (n,mode,reps,dx,v,inputs,raw,outs,cons,q,flux,zero)in states
  foreach(a->copyto!(a,zero),flux)
  for stage in 1:3;launch(stage,flux,cons,q,dx,n);CUDA.synchronize(CUDA.context());end
 end
 for trial in 1:5
  payload=copy(f);elapsed=0.;times=Float64[];correct=true;err=0.;oi=1
  for (n,mode,reps,dx,v,inputs,raw,outs,cons,q,flux,zero)in states
   foreach(a->fill!(a,SENT),outs);ct=zeros(3)
   for rep in 1:reps
    foreach(a->copyto!(a,zero),flux);CUDA.synchronize(CUDA.context())
    for stage in 1:3
     start=time_ns();launch(stage,flux,cons,q,dx,n);CUDA.synchronize(CUDA.context());ct[stage]+=(time_ns()-start)*1e-9
    end
   end
   ct./=reps;append!(times,ct);elapsed+=sum(ct)
   for j in 1:8
    back=Array(raw[j]);correct&=reinterpret(UInt64,back)==reinterpret(UInt64,inputs[j]);append!(payload,back)
   end
   for a in outs
    back=Array(a);correct&=all(x==SENT for x in back[1:4])&&all(x==SENT for x in back[end-3:end]);got=back[5:end-4];ref=oracle[oi:oi+v-1];oi+=v;errors=abs.(got.-ref);err=max(err,maximum(errors));correct&=all(isfinite,got)&&all(errors.<=1e-12.+1e-10.*abs.(ref))
    if mode==0;correct&=all(iszero,got);end
    for k in 0:n-1,j in 0:n-1,i in 0:n-1
     if i<4||j<4||k<4||i>n-5||j>n-5||k>n-5;correct&=got[(k*n+j)*n+i+1]==0.;end
    end
    append!(payload,back)
   end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",correct,",\"checked_values\":",length(payload),",\"max_abs_error\":",err,",\"timing_region\":\"sum9_hypterm_stage_means_complete\",\"jit_excluded\":true,\"internal_repeats\":10,\"timed_kernel_calls\":90,\"case_stage_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  correct||error("FullFluxDivergenceIndependentOracleReadonlyGuardsHalos")
 end
end
