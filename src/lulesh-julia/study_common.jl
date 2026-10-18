using Printf
const G=64
const FIELD_NAMES=(:x,:y,:z,:fx,:fy,:fz,:xd,:yd,:zd,:xdd,:ydd,:zdd,:nodelist,:determ,:fx_elem,:fy_elem,:fz_elem,:dvdx,:dvdy,:dvdz,:x8n,:y8n,:z8n,:sigxx,:sigyy,:sigzz,:delv_xi,:delx_xi,:delv_eta,:delx_eta,:delv_zeta,:delx_zeta,:p,:q,:volo,:v,:vol_error,:nodeElemStart,:nodeElemCornerList,:gamma,:ss,:elemMass,:nodalMass,:symmX,:symmY,:symmZ,:vdov,:delv,:arealg,:dxx,:dyy,:dzz,:vnew,:lzetam,:lzetap,:letap,:letam,:lxip,:lxim,:elemBC,:ql,:qq,:e,:elemRep,:elemElem,)
const FIELD_TYPES=(Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int32,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int32,Int32,Int32,Float64,Float64,Float64,Float64,Int32,Int32,Int32,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Float64,Float64,Float64,Int32,Int32,)
const CASES=[
(nx=1,steps=1,mode=0,lengths=(8,8,8,8,8,8,8,8,8,8,8,8,8,1,8,8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,8,32,1,1,8,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,)),
(nx=3,steps=4,mode=0,lengths=(64,64,64,64,64,64,64,64,64,64,64,64,216,27,216,216,216,216,216,216,216,216,216,27,27,27,27,27,27,27,27,27,27,27,27,27,1,65,216,32,27,27,64,16,16,16,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,)),
(nx=7,steps=4,mode=0,lengths=(512,512,512,512,512,512,512,512,512,512,512,512,2744,343,2744,2744,2744,2744,2744,2744,2744,2744,2744,343,343,343,343,343,343,343,343,343,343,343,343,343,1,513,2744,32,343,343,512,64,64,64,343,343,343,343,343,343,343,343,343,343,343,343,343,343,343,343,343,343,343,)),
(nx=30,steps=3,mode=0,lengths=(29791,29791,29791,29791,29791,29791,29791,29791,29791,29791,29791,29791,216000,27000,216000,216000,216000,216000,216000,216000,216000,216000,216000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,1,29792,216000,32,27000,27000,29791,961,961,961,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,27000,)),
(nx=3,steps=1,mode=1,lengths=(64,64,64,64,64,64,64,64,64,64,64,64,216,27,216,216,216,216,216,216,216,216,216,27,27,27,27,27,27,27,27,27,27,27,27,27,1,65,216,32,27,27,64,16,16,16,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,)),
(nx=3,steps=1,mode=2,lengths=(64,64,64,64,64,64,64,64,64,64,64,64,216,27,216,216,216,216,216,216,216,216,216,27,27,27,27,27,27,27,27,27,27,27,27,27,1,65,216,32,27,27,64,16,16,16,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,)),
(nx=3,steps=1,mode=3,lengths=(64,64,64,64,64,64,64,64,64,64,64,64,216,27,216,216,216,216,216,216,216,216,216,27,27,27,27,27,27,27,27,27,27,27,27,27,1,65,216,32,27,27,64,16,16,16,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,))]
function prepare_states(allocate)
 states=[]
 for (index,c) in enumerate(CASES)
  h=Any[]
  open(joinpath(@__DIR__,"case$(index-1).bin"))do io
   for (T,n) in zip(FIELD_TYPES,c.lengths)
    guard=T==Float64 ? -987654321.125 : Int32(-987654321)
    a=fill(guard,n+2G);read!(io,view(a,G+1:G+n));push!(h,a)
   end
  end
  raw=map(allocate,h);dev=map(a->view(a,G+1:length(a)-G),raw);b=NamedTuple{FIELD_NAMES}(Tuple(dev))
  p=collect(reinterpret(Float64,read(joinpath(@__DIR__,"params$(index-1).bin"))));r=collect(reinterpret(Int32,read(joinpath(@__DIR__,"regions$(index-1).bin"))))
  push!(states,(c=c,h=h,raw=raw,b=b,p=p,p0=copy(p),regions=r,cycle=Int32[0],vdov=zeros(c.nx^3),arealg=zeros(c.nx^3),ss=zeros(c.nx^3),times=zeros(c.steps)))
 end
 states
end
function reset!(s)
 copyto!(s.p,s.p0);s.cycle[1]=0
 for (a,h)in zip(s.raw,s.h);copyto!(a,h);end
 CUDA.synchronize()
end
function time_increment!(s)
 p=s.p
 targetdt=p[19]-p[25]
 if p[18]<=0.0 && s.cycle[1]!=0
  olddt=p[26];gnewdt=1.0e20
  if p[22]<gnewdt;gnewdt=p[22]/2.0;end
  if p[23]<gnewdt;gnewdt=p[23]*2.0/3.0;end
  newdt=gnewdt;ratio=newdt/olddt
  if ratio>=1.0
   if ratio<p[20];newdt=olddt
   elseif ratio>p[21];newdt=olddt*p[21];end
  end
  if newdt>p[24];newdt=p[24];end
  p[26]=newdt
 end
 if targetdt>p[26] && targetdt<4.0*p[26]/3.0;targetdt=2.0*p[26]/3.0;end
 if targetdt<p[26];p[26]=targetdt;end
 p[25]+=p[26];s.cycle[1]+=Int32(1)
end
function constraints!(s)
 p=s.p;p[22]=1.0e20;p[23]=1.0e20
 off=1
 for r=1:11
  n=Int(s.regions[off]);off+=1;ids=view(s.regions,off:off+n-1);off+=n
  ctmp=p[22];celem=Int32(-1);qqc2=64.0*p[11]*p[11]
  for thread=1:2
   for idx0 in ids
    idx=Int(idx0)+1;dtf=s.ss[idx]*s.ss[idx]
    if s.vdov[idx]<0.0;dtf=dtf+qqc2*s.arealg[idx]*s.arealg[idx]*s.vdov[idx]*s.vdov[idx];end
    dtf=s.arealg[idx]/sqrt(dtf)
    if s.vdov[idx]!=0.0 && dtf<ctmp;ctmp=dtf;celem=idx0;end
   end
  end
  if celem!=-1;p[22]=ctmp;end
  htmp=p[23];helem=Int32(-1)
  for thread=1:2
   for idx0 in ids
    idx=Int(idx0)+1
    if s.vdov[idx]!=0.0
     dt=p[16]/(abs(s.vdov[idx])+1e-20)
     if htmp>dt;htmp=dt;helem=idx0;end
    end
   end
  end
  if helem!=-1;p[23]=htmp;end
 end
end
function capture!(values,s,initial)
 for a in s.raw;append!(values,Float64.(Array(a)));end
 if initial;append!(values,s.p0);append!(values,Float64.(s.regions))
 else
  p=s.p;append!(values,[p[25],p[26],Float64(s.cycle[1]),p[22],p[23]])
 end
end
function pipeline!(s,launch,values,capture_output)
 for step=1:s.c.steps
  CUDA.synchronize();start=time_ns();time_increment!(s);b=s.b;p=s.p;numElem=Int32(s.c.nx^3);numNode=Int32((s.c.nx+1)^3);numNodeBC=Int32((s.c.nx+1)^2)
  deltaTime=p[26];u_cut=p[5];hgcoef=p[6];s1=Int32(0);s2=Int32(0);s3=Int32(0)
  monoq_limiter_mult=p[8]
  monoq_max_slope=p[7]
  qlc_monoq=p[9]
  qqc_monoq=p[10]
  e_cut=p[1]
  p_cut=p[2]
  q_cut=p[3]
  v_cut=p[4]
  eosvmax=p[12]
  eosvmin=p[13]
  pmin=p[14]
  emin=p[15]
  rho0=p[17]
  launch(Val(:fill_sig),numElem,b.sigxx,b.sigyy,b.sigzz,b.p,b.q,numElem)
  launch(Val(:integrateStress),numElem,b.fx_elem,b.fy_elem,b.fz_elem,b.x,b.y,b.z,b.nodelist,b.sigxx,b.sigyy,b.sigzz,b.determ,numElem)
  launch(Val(:acc_final_force),numNode,b.fx_elem,b.fy_elem,b.fz_elem,b.fx,b.fy,b.fz,b.nodeElemStart,b.nodeElemCornerList,numNode)
  copyto!(b.vol_error,Int32[-1])
  launch(Val(:hgc),numElem,b.dvdx,b.dvdy,b.dvdz,b.x8n,b.y8n,b.z8n,b.determ,b.x,b.y,b.z,b.nodelist,b.volo,b.v,b.vol_error,numElem)
  launch(Val(:fb),numElem,b.dvdx,b.dvdy,b.dvdz,b.x8n,b.y8n,b.z8n,b.determ,b.xd,b.yd,b.zd,b.ss,b.elemMass,b.nodelist,b.gamma,b.fx_elem,b.fy_elem,b.fz_elem,hgcoef,numElem)
  launch(Val(:collect_final_force),numNode,b.fx_elem,b.fy_elem,b.fz_elem,b.fx,b.fy,b.fz,b.nodeElemStart,b.nodeElemCornerList,numNode)
  launch(Val(:accelerationForNode),numNode,b.fx,b.fy,b.fz,b.nodalMass,b.xdd,b.ydd,b.zdd,numNode)
  launch(Val(:applyAccelerationBoundaryConditionsForNodes),numNodeBC,b.symmX,b.symmY,b.symmZ,b.xdd,b.ydd,b.zdd,s1,s2,s3,numNodeBC)
  launch(Val(:calcVelocityForNodes),numNode,b.xd,b.yd,b.zd,b.xdd,b.ydd,b.zdd,deltaTime,u_cut,numNode)
  launch(Val(:calcPositionForNodes),numNode,b.x,b.y,b.z,b.xd,b.yd,b.zd,deltaTime,numNode)
  launch(Val(:calcKinematicsForElems),numElem,b.xd,b.yd,b.zd,b.x,b.y,b.z,b.nodelist,b.volo,b.v,b.delv,b.arealg,b.dxx,b.dyy,b.dzz,b.vnew,deltaTime,numElem)
  copyto!(b.vol_error,Int32[-1])
  launch(Val(:calcStrainRates),numElem,b.dxx,b.dyy,b.dzz,b.vnew,b.vdov,b.vol_error,numElem)
  copyto!(s.vdov,b.vdov)
  launch(Val(:calcMonotonicQGradientsForElems),numElem,b.xd,b.yd,b.zd,b.x,b.y,b.z,b.nodelist,b.volo,b.delv_eta,b.delx_eta,b.delv_zeta,b.delx_zeta,b.delv_xi,b.delx_xi,b.vnew,numElem)
  launch(Val(:calcMonotonicQForElems),numElem,b.elemBC,b.elemMass,b.ql,b.qq,b.vdov,b.volo,b.delv_eta,b.delx_eta,b.delv_zeta,b.delx_zeta,b.delv_xi,b.delx_xi,b.lxim,b.lxip,b.lzetam,b.lzetap,b.letap,b.letam,b.vnew,monoq_limiter_mult,monoq_max_slope,qlc_monoq,qqc_monoq,numElem)
  launch(Val(:applyMaterialPropertiesForElems),numElem,b.ql,b.qq,b.delv,b.elemRep,b.elemElem,b.q,b.p,b.e,b.ss,b.v,b.vnew,e_cut,p_cut,q_cut,v_cut,eosvmax,eosvmin,pmin,emin,rho0,numElem)
  copyto!(s.arealg,b.arealg);copyto!(s.ss,b.ss);constraints!(s);CUDA.synchronize();s.times[step]=(time_ns()-start)*1e-9
  if capture_output;capture!(values,s,false);end
 end
end
function study(allocate,launch)
 states=prepare_states(allocate);dummy=Float64[]
 for s in states;pipeline!(s,launch,dummy,false);end
 oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"source_cpu_oracle_f64.bin"))))
 for trial=1:5
  values=Float64[];sizehint!(values,15470556);for s in states;reset!(s);capture!(values,s,true);end
  for s in states;pipeline!(s,launch,values,true);end
  @assert length(values)==length(oracle)
  ok=true;maxerr=0.0
  for i in eachindex(values)
   e=abs(values[i]-oracle[i]);maxerr=max(maxerr,e)
   if !isfinite(values[i]) || e>1e-7+3e-8*abs(oracle[i])
    if ok;println("FIRST mismatch ",i," ",values[i]," ",oracle[i]);end;ok=false
   end
  end
  path="lulesh_trial$(trial).bin";open(path,"w")do io;write(io,values);end
  sec=sum(sum(s.times) for s in states)
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"time_seconds\":%.12g,\"max_error\":%.12g,\"timing_region\":\"lulesh_full15kernel_host_adaptive_cycles_excluding_allocation_capture\",\"jit_excluded\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d}}\n",trial,ok ? "true" : "false",sec,maxerr,path,length(values))
 end
end
