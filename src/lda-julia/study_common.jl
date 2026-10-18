const FSENT=-8192.25f0
const ISENT=Int32(-123456789)
const BSENT=UInt8(0xa5)
function lda_buffer(allocate,::Type{T},values,sentinel) where T
 h=vcat(fill(T(sentinel),4),T.(values),fill(T(sentinel),4));d=allocate(T,length(h));copyto!(d,h);return (host=h,raw=d,view=view(d,5:length(h)-4))
end
function lda_reset(c)
 for j in 4:8
  fill!(c.f[j].raw,FSENT);fill!(c.f[j].view,j==4 ? FSENT : 0f0)
 end
 fill!(c.i[3].raw,ISENT);fill!(c.iv[3],Int32(0));copyto!(c.mv,c.trainmask)
end
function lda_snapshot(c,oracle,oi,payload,phase,train_saved)
 ok=true;backsF=[]
 for j in 1:2
  back=Array(c.i[j].raw);ok&=back==c.i[j].host;append!(payload,Float64.(back))
 end
 mb=Array(c.m.raw);wantmask=phase==0 ? c.trainmask : c.valmask;ok&=mb[5:end-4]==wantmask&&all(x==BSENT for x in mb[1:4])&&all(x==BSENT for x in mb[end-3:end]);append!(payload,Float64.(mb))
 maxerr=0.;maxmass=0.
 for j in 1:8
  back=Array(c.f[j].raw);push!(backsF,back[5:end-4]);ok&=all(x==FSENT for x in back[1:4])&&all(x==FSENT for x in back[end-3:end]);append!(payload,Float64.(back))
  if j<=3
   ok&=reinterpret(UInt32,back)==reinterpret(UInt32,c.f[j].host)
  else
   got=Float64.(back[5:end-4]);ref=oracle[oi:oi+length(got)-1];oi+=length(got);errors=abs.(got.-ref);maxerr=max(maxerr,maximum(errors));ok&=all(isfinite,got)&&all(errors.<=2e-5.+1e-4.*abs.(ref))
  end
 end
 lb=Array(c.i[3].raw);ok&=all(x==ISENT for x in lb[1:4])&&all(x==ISENT for x in lb[end-3:end])&&all(iszero,lb[5:end-4]);append!(payload,Float64.(lb))
 gam,nbeta=backsF[4],backsF[6];alpha=c.f[2].host[5:end-4];ptr=c.i[2].host[5:end-4];cts=c.f[1].host[5:end-4]
 for doc in 0:c.D-1
  count=phase==0 ? sum(Float64.(cts[Int(ptr[doc+1])+1:Int(ptr[doc+2])])) : 0.;mass=sum(Float64.(gam[c.K*doc+1:c.K*(doc+1)]));target=sum(Float64.(alpha))+count;maxmass=max(maxmass,abs(mass-target));ok&=abs(mass-target)<=2e-4
  if phase==1;ok&=reinterpret(UInt32,gam[c.K*doc+1:c.K*(doc+1)])==reinterpret(UInt32,alpha);end
 end
 for word in 0:c.NW-2
  mass=sum(Float64.(nbeta[c.K*word+1:c.K*(word+1)]));target=c.reps*Float64(cts[word+1]);maxmass=max(maxmass,abs(mass-target));ok&=abs(mass-target)<=2e-4
 end
 ok&=all(iszero,nbeta[c.K*(c.NW-1)+1:c.K*c.NW])
 if phase==1
  ok&=reinterpret(UInt32,nbeta)==reinterpret(UInt32,train_saved[6])&&reinterpret(UInt32,backsF[7])==reinterpret(UInt32,train_saved[7])
 end
 return ok,oi,maxerr,maxmass,backsF
end
function study(allocate,launch)
 f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));pos=1;cases=[]
 while pos<=length(f)
  K,NW,D,blocks,iters,reps=Int.(f[pos:pos+5]);pos+=6;alpha=f[pos:pos+K-1];pos+=K;beta=f[pos:pos+NW*K-1];pos+=NW*K;counts=f[pos:pos+NW-1];pos+=NW;cols=f[pos:pos+NW-1];pos+=NW;ptr=f[pos:pos+D];pos+=D+1;trainmask=UInt8.(f[pos:pos+NW-1]);pos+=NW;valmask=UInt8.(f[pos:pos+NW-1]);pos+=NW
  fi=(counts,alpha,beta,fill(Float64(FSENT),D*K),zeros(blocks*K),zeros(NW*K),zeros(blocks),zeros(blocks));fb=Tuple(lda_buffer(allocate,Float32,v,FSENT) for v in fi);ib=(lda_buffer(allocate,Int32,cols,ISENT),lda_buffer(allocate,Int32,ptr,ISENT),lda_buffer(allocate,Int32,zeros(NW),ISENT));mb=lda_buffer(allocate,UInt8,trainmask,BSENT)
  push!(cases,(K=K,NW=NW,D=D,blocks=blocks,iters=iters,reps=reps,f=fb,i=ib,m=mb,fv=Tuple(x.view for x in fb),iv=Tuple(x.view for x in ib),mv=mb.view,trainmask=trainmask,valmask=valmask))
 end
 for c in cases
  lda_reset(c)
  for j in 1:c.reps;launch(c,j==1);end
  CUDA.synchronize(CUDA.context());copyto!(c.mv,c.valmask)
  for j in 1:c.reps;launch(c,c.reps==1);end
  CUDA.synchronize(CUDA.context())
 end
 for trial in 1:5
  payload=copy(f);elapsed=0.;times=Float64[];ok=true;oi=1;err=0.;masserr=0.
  for c in cases
   lda_reset(c);CUDA.synchronize(CUDA.context());start=time_ns()
   for j in 1:c.reps;launch(c,j==1);end
   CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9/c.reps;elapsed+=tm;push!(times,tm);pass,oi,em,mm,trainback=lda_snapshot(c,oracle,oi,payload,0,nothing);ok&=pass;err=max(err,em);masserr=max(masserr,mm)
   copyto!(c.mv,c.valmask);CUDA.synchronize(CUDA.context());start=time_ns()
   for j in 1:c.reps;launch(c,c.reps==1);end
   CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9/c.reps;elapsed+=tm;push!(times,tm);pass,oi,em,mm,_=lda_snapshot(c,oracle,oi,payload,1,trainback);ok&=pass;err=max(err,em);masserr=max(masserr,mm)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",err,",\"max_mass_identity_residual\":",masserr,",\"timing_region\":\"sum6_lda_train_validation_2launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":2,\"timed_kernel_calls\":12,\"case_phase_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullLDAPhaseStatesTrueDigammaOracleMassReadonlyGuards")
 end
end
