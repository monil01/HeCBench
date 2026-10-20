# CPU-only independent matrix/slice implementation. No source CUDA/host derivative routines.
using LinearAlgebra,Printf,SHA
BLAS.set_num_threads(16)
P=abspath(ARGS[1]);M=500;N=1000;R=M-2;C=N-2;steps=1000
raw=reinterpret(Float32,read(joinpath(P,"actual_inputs.bin")));off=0
function take(n)
 global off
 z=Float64.(raw[off+1:off+n]);off+=n;z
end
fdt=take(1)[1];T=reshape(take(N*R),N,R);psi=reshape(take(N*R),N,R)
dsc=reshape(take(R*R),R,R);dsr=reshape(take(C*C),C,C);ei=reshape(take(C*R),C,R);tr=take(M*N);trnu=take(N)
ztop=reshape(take(4N),N,4);zbot=reshape(take(4N),N,4);nutop=reshape(take(4N),N,4);nubot=reshape(take(4N),N,4)
for _=1:9;take(N*R);end
@assert off==length(raw)
native=reinterpret(Float32,read(joinpath(P,"native_prepare_outputs.bin")))
# Source output field order: T,dt,psi,omega,u,v,x,y,z,Gfinal,Dzz,DxT,temp,nutop,nubot.
base=N*R+1+N*R+C*R+9N*R+8N
# Count directly: 12 full N*R fields? T,psi,u,v,x,y,z,Gfinal,Dzz,DxT,temp =11.
base=11N*R+C*R+1+8N
history=Float64.(native[base+1:base+steps]);nusource=Float64.(native[base+steps+1:end-2]);@assert length(native)==base+steps+length(nusource)+2
const DX32=1f0/(500f0-1f0);const DX=Float64(DX32);const DX2=Float64(DX32*DX32)
const CAP32=(DX32*DX32)/4f0;const OC=Float64(-(((DX32*DX32)*DX32)*DX32)*100000000f0)
function dz(f,psiflag)
 y=similar(f);y[:,2:R-1]=(f[:,3:R]-f[:,1:R-2])/(2DX)
 y[:,1]=(f[:,2].-(psiflag ? 0. : 1.))/(2DX);y[:,R]=-f[:,R-1]/(2DX);y
end
function dx(f,psiflag)
 y=zeros(N,R);y[2:N-1,:]=(f[3:N,:]-f[1:N-2,:])/(2DX)
 if psiflag
  y[1,:]=(6f[2,:]-3f[3,:]+(2/3)*f[4,:])/(2DX)
  y[N,:]=(-6f[N-1,:]+3f[N-2,:]-(2/3)*f[N-3,:])/(2DX)
 end
 y
end
function dzz(f)
 y=similar(f);y[:,2:R-1]=(f[:,1:R-2]-2f[:,2:R-1]+f[:,3:R])/DX2
 y[:,1]=(1 .-2f[:,1]+f[:,2])/DX2;y[:,R]=(f[:,R-1]-2f[:,R])/DX2;y
end
function dxx(f)
 y=similar(f);y[2:N-1,:]=(f[1:N-2,:]-2f[2:N-1,:]+f[3:N,:])/DX2
 y[1,:]=(-2f[1,:]+2f[2,:])/DX2;y[N,:]=(-2f[N,:]+2f[N-1,:])/DX2;y
end
function g(f)
 DxT=dx(f,false);omega=DxT[2:N-1,:]*dsc;omega=dsr*omega;omega .*=ei;omega=omega*dsc;omega=dsr*omega;omega .*=OC
 psi=zeros(N,R);psi[2:N-1,:]=omega
 u=dz(psi,true);v=-dx(psi,true);dt=min(DX/max(maximum(abs,u),maximum(abs,v)),Float64(CAP32))
 buff=dzz(f);y=dxx(f)+buff+u.*DxT;u=dz(f,false).*v;y .+=u
 (;out=y,u,v,psi,omega,buff,DxT,dt)
end
nu=Float64[];frame=0f0;tstep=0;adaptive_max_ratio=0.;dt=fdt;start=time();x=y=z=nothing;temp=copy(T)
for step=1:steps
 x=g(T);err=abs(x.dt-history[step]);global adaptive_max_ratio=max(adaptive_max_ratio,err/(.003*max(abs(x.dt),1e-12)))
 a=Float64(Float32(Float32(dt)/3f0));temp=T+a*x.out;y=g(temp)
 a=Float64(2f0*(Float32(dt)/3f0));temp=T+a*y.out;z=g(temp)
 a=Float32(dt)/4f0;T .+=Float64(a)*x.out;T .+=Float64(3f0*a)*z.out
 dt=history[step];frame=Float32(frame+Float32(dt))
 if frame>CAP32
  global tstep+=1
  nutop[:,1:3]=T[:,R-2:R];nutop[:,4].=0.;nutop .+=ztop;nutop .-=1.
  nutop[:,1]=(-2/3)*nutop[:,1]+3nutop[:,2]-6nutop[:,3]+(11/3)*nutop[:,4]
  nutop .*=Float64(1f0/(2f0*DX32));nutop[:,1] .*=trnu;nutop[:,2].=1.;push!(nu,sum(nutop[:,1])/(-2))
  nubot[:,1].=1.;nubot[:,2:4]=T[:,1:3];nubot .+=zbot;nubot .-=1.
  nubot[:,1]=(-11/3)*nubot[:,1]+6nubot[:,2]-3nubot[:,3]+(2/3)*nubot[:,4]
  nubot .*=Float64(1f0/(2f0*DX32));nubot[:,1] .*=trnu
  global frame=0f0
 end
 if step%50==0;@printf("CPU oracle step=%d/1000 elapsed=%.3fs Tmax=%.12g\n",step,time()-start,maximum(abs,T));flush(stdout);end
 global dt,T,x,y,z,temp,frame
end
fields=[("T",vec(T)),("dt",[history[end]]),("psi",vec(z.psi)),("omega",vec(z.omega)),("u",vec(z.u)),("v",vec(z.v)),("xrk3",vec(x.out)),("yrk3",vec(y.out)),("zrk3",vec(z.out)),("y",vec(z.out)),("Tbuff",vec(z.buff)),("DxT",vec(z.DxT)),("temp",vec(temp)),("nutop",vec(nutop)),("nubot",vec(nubot)),("dt_history",history),("Nu_history",nu),("frames",[Float64(frame)]),("tstep",[Float64(tstep)])]
gold=Float64[];budgets=Float64[];records=String[];off=0;allpass=true
for (name,v) in fields
 scale=max(maximum(abs,v;init=0.),name in ["dt","dt_history","frames"] ? 1e-12 : 1.)
 budget=.003*scale;name=="tstep"&&(budget=1e-12)
 got=Float64.(native[off+1:off+length(v)]);err=maximum(abs.(got-v);init=0.);ratio=err/budget
 push!(records,"{\"name\":\"$name\",\"count\":$(length(v)),\"max_abs_error\":$err,\"scale\":$scale,\"budget\":$budget,\"max_budget_ratio\":$ratio}")
 global allpass &= all(isfinite,got)&&all(isfinite,v)&&ratio<=1.;append!(gold,v);append!(budgets,fill(budget,length(v)));global off+=length(v)
end
@assert off==length(native);allpass &= adaptive_max_ratio<=1.
write(joinpath(P,"independent_oracle.bin"),gold);write(joinpath(P,"oracle_budgets.bin"),budgets)
open(joinpath(P,"independent_oracle_report.json"),"w") do f
 print(f,"{\"pass\":",allpass," ,\"CPU_seconds\":",time()-start,",\"adaptive_dt_max_budget_ratio\":",adaptive_max_ratio,",\"steps\":1000,\"Nu_samples\":",length(nu),",\"fields\":[",join(records,","),"],\"scope\":\"Float64 full source-shaped RK3 driven by independently checked actual adaptive step history; source rounded coefficients/data retained; engineering field norm budgets, not continuum convergence\"}\n")
end
@assert allpass "Native full trajectory failed predeclared independent CPU field budget"
