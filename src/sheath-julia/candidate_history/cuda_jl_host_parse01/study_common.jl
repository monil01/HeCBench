using Libdl,Printf
const ROOT=normpath(@__DIR__)
const CHECKLIB=Libdl.dlopen(joinpath(pwd(),"studyphysics.so"))
const DEPOSITION=Libdl.dlsym(CHECKLIB,:witness_deposition)
const FIELD=Libdl.dlsym(CHECKLIB,:witness_field)
const MOTION=Libdl.dlsym(CHECKLIB,:witness_motion)
const REPORT=Libdl.dlsym(CHECKLIB,:witness_report)
const K=1.38065e-23;const EV=11604.52;const Q=1.602176565e-19;const ME=9.10938215e-31;const AMU=1.660538921e-27
const DX=1e-4;const XL=100*DX;const EPS=8.85418782e-12
function guarded(::Type{T},n) where T
 raw=CuArray{UInt8}(undef,n*sizeof(T)+512);fill!(raw,UInt8(0xa5));d=unsafe_wrap(CuArray,convert(CuPtr{T},pointer(raw)+256),n;own=false);CUDA.memset(convert(CuPtr{UInt8},pointer(d)),UInt8(0),n*sizeof(T))
 (raw=raw,d=d,n=n)
end
function bytes(a)
 v=Array(a.raw);all(==(0xa5),v[1:256])&&all(==(0xa5),v[end-255:end])||error("actual guard bytes")
 v
end
function deposition_check(before,n,raw,scaled,weight)
 ccall(DEPOSITION,Cint,(Ptr{PICParticle},Cint,Ptr{Float32},Ptr{Float32},Cdouble),before,n,raw,scaled,weight)==0||error("independent deposition")
 nothing
end
function motion_check(before,after,n,ef,qm,rewind)
 ccall(MOTION,Cint,(Ptr{PICParticle},Ptr{PICParticle},Cint,Ptr{Float64},Cdouble,Cint),before,after,n,ef,qm,rewind)==0||error("independent particle motion")
 nothing
end
function solve_potential!(phi,rho)
 phi[1]=phi[101]=0.0;dx2=DX*DX
 @inbounds for it in 0:39999
  for i in 2:100
   g=.5*(phi[i-1]+phi[i+1]+dx2*rho[i]/EPS);phi[i]=phi[i]+1.4*(g-phi[i])
  end
  if rem(it,25)==0
   sum=0.0
   for i in 2:100;R=-rho[i]/EPS-(phi[i-1]-2*phi[i]+phi[i+1])/dx2;sum+=R*R;end
   sqrt(sum)/101<1e-4&&return true
  end
 end
 println("Gauss-Seidel solver failed to converge")
 false
end
function scatter_species!(parts,den,host,n,weight)
 CUDA.memset(convert(CuPtr{UInt8},pointer(den.d)),UInt8(0),101*4);CUDA.device_synchronize();start=time_ns();scatter_launch!(parts.d,den.d,n);CUDA.device_synchronize();kernel_seconds=(time_ns()-start)*1e-9;copyto!(host,den.d);factor=weight/DX
 @inbounds for i in 1:101;host[i]=Float32(Float64(host[i])*factor);end
 host[1]=Float32(Float64(host[1])*2.0);host[101]=Float32(Float64(host[101])*2.0)
 kernel_seconds
end
function write_grid(io,ts,ni,ne,rho,phi,ef)
 @printf(io,"ZONE I=%d T=ZONE_%06d\n",101,ts)
 for i in 1:101;@printf(io,"%g %g %g %g %g %g\n",(i-1)*DX,ne[i],ni[i],rho[i],phi[i],ef[i]);end
 flush(io)
 nothing
end
function study()
 CUDA.allowscalar(false);phiscale=K*(3*EV)/Q;escale=phiscale/DX;rhoscale=Q*1e16;phiunits=collect(reinterpret(Float64,read(joinpath(ROOT,"data/case0_phi_normalization_v2.bin"))))
 length(phiunits)==101&&all(x->isfinite(x)&&x>0,phiunits)||error("phi units")
 rawdir=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/sheath_primary",MODEL));mkpath(rawdir);capture=string(time_ns())
 for trial in 0:5
  seconds=0.0;prefix=Float64[];out=Float64[];rawpath=joinpath(rawdir,"$(capture)_trial$(trial).physical.bin");raw=trial>0 ? open(rawpath,"w") : nothing
  trial>0&&println("SHEATH_RAW trial=",trial," path=",rawpath)
  for ci in 0:2
   fixture=read(joinpath(ROOT,"data/case$(ci).bin"));header=reinterpret(Int32,fixture[1:12]);n=Int(header[1]);steps=Int(header[2]);cold=Int(header[3]);values=reinterpret(Float64,fixture[13:end]);length(values)==4n||error("particle fixture size")
   hi=[PICParticle(values[2i-1],values[2i],true)for i in 1:n];he=[PICParticle(values[2n+2i-1],values[2n+2i],true)for i in 1:n]
   phi=zeros(Float64,101);rho=similar(phi);ef=similar(phi);ni=zeros(Float32,101);ne=similar(ni)
   pi=guarded(PICParticle,n);pe=guarded(PICParticle,n);di=guarded(Float32,101);de=guarded(Float32,101);df=guarded(Float64,101);buffers=(pi,pe,di,de,df);copyto!(pi.d,hi);copyto!(pe.d,he);CUDA.device_synchronize();init_i=Array(pi.d);init_e=Array(pe.d)
   push!(prefix,Float64(ci),Float64(n),Float64(steps))
   for (actual,want)in ((init_i,hi),(init_e,he))
    for i in 1:n
     a=actual[i];b=want[i];a.x===b.x&&a.v===b.v&&a.alive==b.alive||error("actual initial readback");push!(prefix,a.x,a.v,a.alive ? 1.0 : 0.0)
    end
   end
   weight=1e16*XL/n;qmi=Q/(16*AMU);qme=-Q/ME;sp_time=0.0
   function snapshot(stage,step)
    trial==0&&return
    write(raw,Int32[ci,stage,step,5])
    for a in buffers;v=bytes(a);write(raw,UInt64(length(v)));write(raw,v);end
    nothing
   end
   function timed(fn)
    start=time_ns();fn();seconds+=(time_ns()-start)*1e-9
    nothing
   end
   function scattercheck()
    deposition_check(hi,n,Array(di.d),ni,weight);deposition_check(he,n,Array(de.d),ne,weight)
    nothing
   end
   function fields(active)
    converged=false
    function body()
     @inbounds for i in 1:101;rho[i]=Q*Float64(ni[i])+(-Q)*Float64(ne[i]);end
     converged=solve_potential!(phi,rho)
     @inbounds for i in 2:100;ef[i]=-(phi[i+1]-phi[i-1])/(2*DX);end
     ef[1]=-(phi[2]-phi[1])/DX;ef[101]=-(phi[101]-phi[100])/DX;copyto!(df.d,ef);CUDA.device_synchronize()
     nothing
    end
    if active;timed(body);else;body();end
    ccall(FIELD,Cint,(Ptr{Float32},Ptr{Float32},Ptr{Float64},Ptr{Float64},Ptr{Float64},Cint),ni,ne,rho,phi,ef,converged)==0||error("independent Poisson/field")
    Array(df.d)==ef||error("actual GPU field input")
    if trial>0
     write(raw,Int32[ci,1,active ? 1 : 0,-1]);for a in (ni,ne,rho,phi,ef);write(raw,a);end
    end
    @inbounds for j in 1:101;push!(out,Float64(ni[j])/1e16,Float64(ne[j])/1e16,rho[j]/rhoscale,phi[j]/(ci==0 ? phiunits[j] : phiscale),ef[j]/escale,converged ? 1.0 : 0.0);end
    nothing
   end
   snapshot(0,0);sp_time+=scatter_species!(pi,di,ni,n,weight);sp_time+=scatter_species!(pe,de,ne,n,weight);scattercheck();fields(false);rewind_launch!(pi.d,df.d,qmi,n);rewind_launch!(pe.d,df.d,qme,n);CUDA.device_synchronize();afteri=Array(pi.d);aftere=Array(pe.d);motion_check(hi,afteri,n,ef,qmi,1);motion_check(he,aftere,n,ef,qme,1);hi=afteri;he=aftere;snapshot(2,0)
   io=open(joinpath(rawdir,"$(capture)_trial$(trial)_case$(ci).dat"),"w");println(io,"VARIABLES = x nde ndi rho phi ef");write_grid(io,0,ni,ne,rho,phi,ef)
   for ts in 1:steps
    timed()do
     sp_time+=scatter_species!(pi,di,ni,n,weight);sp_time+=scatter_species!(pe,de,ne,n,weight)
    end
    scattercheck();fields(true)
    timed()do
     push_launch!(pe.d,df.d,qme,n);push_launch!(pi.d,df.d,qmi,n);CUDA.device_synchronize()
    end
    afteri=Array(pi.d);aftere=Array(pe.d);motion_check(hi,afteri,n,ef,qmi,0);motion_check(he,aftere,n,ef,qme,0);hi=afteri;he=aftere;snapshot(3,ts)
    timed()do
     if rem(ts,25)==0
      maxphi=abs(phi[1]);for i in 1:101;abs(phi[i])>maxphi&&(maxphi=abs(phi[i]));end
      @printf("TS:%i\tnp_i:%d\tnp_e:%d\tdphi:%.3g\n",ts,n,n,maxphi-phi[1])
     end
     rem(ts,1000)==0&&write_grid(io,ts,ni,ne,rho,phi,ef)
    end
   end
   close(io);vthi=sqrt(2*K*1.0*EV/(16*AMU));vthe=sqrt(2*K*3.0*EV/ME)
   for (parts,vth)in ((hi,vthi),(he,vthe));for a in parts;push!(out,a.x/XL,a.v/vth,a.alive ? 1.0 : 0.0);end;end
   for a in buffers;v=bytes(a);append!(out,Float64.(v[1:256]));append!(out,Float64.(v[end-255:end]));end
   trial>0&&println("SHEATH_CASE trial=",trial," case=",ci," n=",n," steps=",steps," cold=",cold," scatter_kernel_seconds=",sp_time," alive_i=",count(p->p.alive,hi)," alive_e=",count(p->p.alive,he))
   for a in buffers;CUDA.unsafe_free!(a.d);CUDA.unsafe_free!(a.raw);end
  end
  if trial>0
   close(raw);append!(prefix,out);length(prefix)==6683535||error("complete primary length");path="trial$(trial).bin";open(io->write(io,prefix),path,"w");ccall(REPORT,Cvoid,())
   println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":",length(prefix),",\"max_abs_error\":null,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum3_fullPICmainloops_observerpaused_contextsplit\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(prefix),",\"byte_order\":\"little\"}}")
  end
 end
end
