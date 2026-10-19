using CUDA,Printf
include(joinpath(@__DIR__,"study_params.jl"))
function amg_read(name,::Type{T},n)where T
 bytes=read(joinpath(@__DIR__,name));length(bytes)==n*sizeof(T)||error("assetbytes $name");collect(reinterpret(T,bytes))
end
amg_pad(a,g)=vcat(fill(g,16),a,fill(g,16))
amg_guards(a,g)=all(==(g),a[1:16])&&all(==(g),a[end-15:end])
function host_matvec!(values,idx,cols,x,y)
 # OriginalactiveHYPREalpha1/beta0,singlevector/fullrowbranch; CPUexecutionretained.
 fill!(y,0.0)
 @inbounds for i in eachindex(y)
  temp=y[i]
  for j in Int(idx[i])+1:Int(idx[i+1]);temp+=values[j]*x[Int(cols[j])+1];end
  y[i]=temp
 end
 nothing
end
function host_axpy!(x,y)
 @inbounds for i in eachindex(y);y[i]+=.5*x[i];end
 nothing
end
function host_matvec_batch!(V,I,J,x,y)
 for _ in 1:R;host_matvec!(V,I,J,x,y);end
end
function host_axpy_batch!(x,y)
 for _ in 1:R;host_axpy!(x,y);end
end
function gpu_relax_batch!(launch,device_args)
 for _ in 1:R;launch(device_args...);end
 CUDA.device_synchronize(blocking=true)
end
function amg_metrics(u,w)
 finite=all(isfinite,u);range=all(v->0<=v<=2,view(u,17:N+16));unweighted=0.0;weighted=0.0;residual=0.0
 @inbounds for i in 0:N-1
  a=u[i+17];unweighted=max(unweighted,abs(a-1));weighted=max(weighted,abs(a-1)/w[i+1]);x=i%NX;y=(i÷NX)%NY;z=i÷(NX*NY);neighbors=0.0;deg=0
  if z>0;neighbors+=u[i-NX*NY+17];deg+=1;end
  if y>0;neighbors+=u[i-NX+17];deg+=1;end
  if x>0;neighbors+=u[i-1+17];deg+=1;end
  if x+1<NX;neighbors+=u[i+1+17];deg+=1;end
  if y+1<NY;neighbors+=u[i+NX+17];deg+=1;end
  if z+1<NZ;neighbors+=u[i+NX*NY+17];deg+=1;end
  residual=max(residual,abs(6*a-neighbors-(6-deg))/w[i+1])
 end
 (finite=finite,range=range,unweighted=unweighted,weighted=weighted,residual=residual)
end
function amg_study(launch,observer,model;allocate=CuArray)
 I=amg_read("csr_i_i32.bin",Int32,N+1);J=amg_read("csr_j_i32.bin",Int32,E);V=amg_read("csr_values_f64.bin",Float64,E);F=amg_read("rhs_f64.bin",Float64,N);w=amg_read("weights_f64.bin",Float64,N)
 ip=amg_pad(I,Int32(-123456789));jp=amg_pad(J,Int32(-123456789));vp=amg_pad(V,91.25);fp=amg_pad(F,91.25);u0=amg_pad(amg_read("relax_u0_f64.bin",Float64,N),91.25);mx=amg_pad(amg_read("matvec_x_f64.bin",Float64,N),91.25);ax=amg_pad(amg_read("axpy_x_f64.bin",Float64,N),91.25);ay0=amg_pad(amg_read("axpy_y0_f64.bin",Float64,N),91.25);my=amg_pad(zeros(N),91.25);ay=copy(ay0);expectedM=amg_read("matvec_expected_f64.bin",Float64,N);expectedA=amg_read("axpy_expected_f64.bin",Float64,N)
 di=allocate(ip);dj=allocate(jp);dv=allocate(vp);df=allocate(fp);du=allocate(u0);params=allocate(zeros(Int32,5));device_args=(view(dv,17:E+16),view(di,17:N+17),view(dj,17:E+16),view(du,17:N+16),view(df,17:N+16),Int32(N));xm=view(mx,17:N+16);ym=view(my,17:N+16);xa=view(ax,17:N+16);ya=view(ay,17:N+16)
 rawdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","amgmk-julia-shared","raw",model);mkpath(rawdir)
 dump(a,name)=begin;p=joinpath(rawdir,"$(name)_pid$(getpid()).bin");write(p,a);abspath(p);end
 eq=amg_pad(ones(N),91.25);copyto!(du,eq);gpu_relax_batch!(launch,device_args);eqgot=Array(du);eqok=eqgot==eq;eqpath=dump(eqgot,"equilibrium");eqok||error("sourceequilibriumcontrolfailed")
 host_matvec_batch!(V,I,J,xm,ym);host_axpy_batch!(xa,ya);copyto!(du,u0);gpu_relax_batch!(launch,device_args)
 for trial in 1:5
  fill!(ym,0.0);copyto!(ay,ay0);copyto!(du,u0);observer(params,Int32(N));CUDA.device_synchronize(blocking=true)
  pre=Float64[];sizehint!(pre,COUNT);append!(pre,Float64[NX,NY,NZ,N,E,R,.5,QBOUND,WMIN,M0UPPER,ROUND_BUDGET,WEIGHTED_BOUND,EVAL_MARGIN]);echo=Array(params);append!(pre,Float64.(echo));append!(pre,Float64.(I));append!(pre,Float64.(J));append!(pre,V)
  ib=Array(di);jb=Array(dj);vb=Array(dv);fb=Array(df);ub=Array(du)
  for a in (ib,jb,vb,fb,ub,mx,ax,my,ay,w);append!(pre,Float64.(a));end
  t=time_ns();host_matvec_batch!(V,I,J,xm,ym);mt=(time_ns()-t)*1e-9
  t=time_ns();gpu_relax_batch!(launch,device_args);gt=(time_ns()-t)*1e-9
  t=time_ns();host_axpy_batch!(xa,ya);at=(time_ns()-t)*1e-9
  iend=Array(di);jend=Array(dj);vend=Array(dv);fend=Array(df);u=Array(du)
  for a in (I,J,V,iend,jend,vend,fend,mx,ax);append!(pre,Float64.(a));end
  length(pre)==PREFIX||error("prefixcount$(length(pre))")
  inputs=iend==ib&&jend==jb&&vend==vb&&fend==fb&&ub==u0;echo_ok=echo==Int32[N,NX,NY,NZ,R];z=amg_metrics(u,w);cpu=ym==expectedM&&ya==expectedA&&all(a->amg_guards(a,91.25),(mx,my,ax,ay));gg=amg_guards(u,91.25)
  flags=Int[z.finite,z.range,z.unweighted<=E0MAX+1e-10,z.weighted<=WEIGHTED_BOUND,z.residual<=RESID_BOUND,inputs,echo_ok,eqok];ok=cpu&&gg&&all(==(1),flags)
  append!(pre,my);append!(pre,ay);append!(pre,u[1:16]);append!(pre,u[end-15:end]);append!(pre,Float64.(flags));length(pre)==COUNT||error("outputcount")
  rp=dump(u,"trial$(trial)");out=joinpath(ENV["STUDY_OUTPUT_DIR"],"amgmk_scores_trial$(trial).bin");write(out,pre)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"host_numeric_max_abs_error\":%s,\"jit_excluded\":true,\"timing_region\":\"source_CPUmatvec_GPUasyncRelax_CPUaxpy_R500_elapsed_sum\",\"internal_repeats\":500,\"verification_contract\":\"native_property_scores\",\"numerical_agreement_claim\":false,\"raw_output_identity_claim\":false,\"component_time_seconds\":[%.17g,%.17g,%.17g],\"relax_weighted_error\":%.17g,\"relax_weighted_bound\":%.17g,\"relax_residual_weighted_max\":%.17g,\"relax_residual_weighted_bound\":%.17g,\"raw_values_checked\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"raw_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d},\"equilibrium_artifact\":{\"path\":\"%s\",\"count\":%d}}\n",trial,mt+gt+at,ok ? "true" : "false",length(pre),cpu ? "0" : "null",mt,gt,at,z.weighted,WEIGHTED_BOUND,z.residual,RESID_BOUND,N,out,length(pre),rp,N+32,eqpath,N+32);flush(stdout);ok||error("fullstateproperty/hostoperatorcheckfailed")
 end
end
