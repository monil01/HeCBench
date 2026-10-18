const SENT= -777.0
function loadf(name)
 p=joinpath(@__DIR__,name);filesize(p)%8==0||error("file size")
 open(io->read!(io,Vector{Float64}(undef,filesize(p)÷8)),p)
end
sentinel(::Type{Int32})=Int32(SENT)
sentinel(::Type{Float64})=SENT
sentinel(::Type{RSC})=RSC(SENT,SENT)
sentinel(::Type{RSPole})=RSPole(ntuple(_->sentinel(RSC),4)...,Int16(SENT))
sentinel(::Type{RSWindow})=RSWindow(SENT,SENT,SENT,Int32(SENT),Int32(SENT))
flat(v::Vector{T})where{T<:Union{Int32,Float64}}=Float64.(v)
flat(v::Vector{RSPole})=Float64[z for x in v for z in (x.ea.r,x.ea.i,x.rt.r,x.rt.i,x.ra.r,x.ra.i,x.rf.r,x.rf.i,x.l)]
flat(v::Vector{RSWindow})=Float64[z for x in v for z in (x.t,x.a,x.f,x.start,x.stop)]
function study(allocate,launch)
 f=loadf("fixtures_f64.bin");oracle=loadf("oracle_f64.bin");nc,maxp,maxw,maxn,ncases=Int.(f[1:5]);header=f[1:5+2ncases];pos=length(header)+1
 function take(n)
  a=f[pos:pos+n-1];pos+=n;a
 end
 num=Int32.(take(12));mats=Int32.(take(12maxn));nwin=Int32.(take(nc));concs=take(12maxn);pseudo=take(4nc)
 pv=take(nc*maxp*9);poles=[RSPole(RSC(pv[i],pv[i+1]),RSC(pv[i+2],pv[i+3]),RSC(pv[i+4],pv[i+5]),RSC(pv[i+6],pv[i+7]),Int16(pv[i+8])) for i in 1:9:length(pv)]
 wv=take(nc*maxw*5);windows=[RSWindow(wv[i],wv[i+1],wv[i+2],Int32(wv[i+3]),Int32(wv[i+4])) for i in 1:5:length(wv)]
 pos==length(f)+1||error("trailing fixture")
 host=(num,mats,nwin,concs,pseudo,poles,windows);bases=map(a->allocate(vcat([sentinel(eltype(a))],a,[sentinel(eltype(a))])),host);arrays=map(b->view(b,2:length(b)-1),bases);numd,matsd,nwind,concsd,pseudod,polesd,windowsd=arrays
 function readfixture()
  values=copy(header)
  for a in arrays;append!(values,flat(Array(a)));end
  values
 end
 cases=Any[]
 for ci in 1:ncases
  n,doppler=Int.(header[4+2ci:5+2ci]);outbase=allocate(fill(Int32(SENT),n+2));detailbase=allocate(fill(SENT,8n+2));push!(cases,(n=n,doppler=doppler,outbase=outbase,detailbase=detailbase,out=view(outbase,2:n+1),details=view(detailbase,2:8n+1)))
 end
 function run(c,diagnostic)
  launch(numd,concsd,matsd,nwind,pseudod,windowsd,polesd,c.out,c.details,c.n,c.doppler,maxw,maxp,maxn,diagnostic)
  CUDA.device_synchronize(blocking=true)
 end
 for c in cases;run(c,Val(false));run(c,Val(true));end
 for trial in 1:5
  prefix=readfixture();ok=prefix==f;payload=copy(prefix);total=0.;times=Float64[];opos=1;maxabs=0.;maxbudget=0.
  for c in cases
   fill!(c.outbase,Int32(SENT));fill!(c.detailbase,SENT);CUDA.device_synchronize(blocking=true)
   start=time_ns();run(c,Val(false));elapsed=(time_ns()-start)*1e-9;total+=elapsed;push!(times,elapsed)
   actualout=Array(c.out);run(c,Val(true));details=Array(c.details)
   for i in 1:c.n
    row=vcat(Float64(actualout[i]),details[8(i-1)+1:8i]);ref=oracle[opos:opos+8];opos+=9
    ok &=row[1:5]==ref[1:5] && all(isfinite,row)
    for j in 6:9
     err=abs(row[j]-ref[j]);budget=1e-9+1e-8*abs(ref[j]);maxabs=max(maxabs,err);maxbudget=max(maxbudget,err/budget);ok &=err<=budget
    end
    append!(payload,row)
   end
   guard=vcat(Float64.(Array(c.outbase)[[1,end]]),Array(c.detailbase)[[1,end]]);ok &=all(==(SENT),guard);append!(payload,guard)
  end
  readonly=readfixture();ok &=readonly==f;append!(payload,readonly)
  for b in bases
   a=Array(b);guards=flat(a[[1,end]]);ok &=all(==(SENT),guards);append!(payload,guards)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",maxabs,",\"max_rel_error\":null,\"max_normalized_oracle_budget\":",maxbudget,",\"timing_region\":\"sum8_source_event_lookup_kernels_context_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":8,\"untimed_diagnostic_kernel_calls\":8,\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("IndependentfullXS/argmax/RNG/readonly/guards failed")
 end
end
