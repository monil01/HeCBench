# Original GPUQT host solver equations,Julia translation Codex2026-10-19.
# Copyright2017 Zheyong Fan,Ville Vierimaa,Ari Harju;GPLv3-or-later.
using Printf
json(x::AbstractString)="\""*replace(x,'\\'=>"\\\\",'"'=>"\\\"")*"\""
json(x::Bool)=x ? "true" : "false"
json(x::Nothing)="null"
json(x::Number)=isfinite(x) ? string(x) : "null"
json(x::AbstractVector)="["*join(json.(x),",")*"]"
json(x::AbstractDict)="{"*join((json(string(k))*":"*json(v)for(k,v)in x),",")*"}"
mutable struct StudyState
 capture::Bool;roi::Bool;paused::Int;mark::UInt64;seconds::Float64;valid::Bool
 trial::Int;case_id::Int;r::Int;kind::Int;step::Int;orbital::Int;next_id::Int
 prefix::Vector{Float64};output::Vector{Float64};released::Dict{Int,Vector{Float64}};live::Dict{Int,Any}
 launches::Vector{Int};timed_launches::Vector{Int};raw::IO;guards::IO;layout::IO;work::IO
end
StudyState()=StudyState(false,false,0,UInt64(0),0.0,true,0,0,0,0,0,-1,0,Float64[],Float64[],Dict{Int,Vector{Float64}}(),Dict{Int,Any}(),zeros(Int,14),zeros(Int,14),devnull,devnull,devnull,devnull)
function pause!(s)
 if s.roi
  s.paused+=1;if s.paused==1;s.seconds+=(time_ns()-s.mark)*1e-9;end
 end
end
function resume!(s)
 if s.roi
  s.paused-=1;if s.paused==0;s.mark=time_ns();end
 end
end
function paused(f,s)
 pause!(s);try;f();finally;resume!(s);end
end
function timed(f,s,kind)
 CUDA.device_synchronize();@assert !s.roi;s.kind=kind;s.step=0;s.paused=0;s.roi=true;s.mark=time_ns()
 f();CUDA.device_synchronize();@assert s.paused==0;s.seconds+=(time_ns()-s.mark)*1e-9;s.roi=false
end
function record_launch!(s,code)
 s.launches[code+1]+=1;if s.roi;s.timed_launches[code+1]+=1;end;nothing
end
function record_scalar!(s,code,values)
 s.capture||return;append!(s.prefix,(code,s.case_id,s.kind,s.r,s.step,s.orbital,length(values)));append!(s.prefix,values)
end
struct Buffer{T,F,D}
 id::Int;full::F;data::D;count::Int
end
function allocate!(s,::Type{T},n)where T
 n=Int(n);g=256÷sizeof(T);full=CuArray{T}(undef,n+2*g);CUDA.device_synchronize()
 paused(s)do
  host=reinterpret(T,fill(UInt8(165),(n+2*g)*sizeof(T)));copyto!(full,host);CUDA.device_synchronize()
 end
 data=@view full[g+1:g+n];id=s.next_id;s.next_id+=1;b=Buffer{T,typeof(full),typeof(data)}(id,full,data,n);s.live[id]=b;b
end
function snapshot!(s,b::Buffer{T},event)where T
 host=Array(b.full);physical=collect(reinterpret(UInt8,host));N=b.count*sizeof(T);first=physical[1:256];last=physical[end-255:end];s.valid&=all(==(165),first)&&all(==(165),last)
 if s.capture
  row=Dict("kind"=>"gpu_buffer","event"=>event,"id"=>b.id,"dtype"=>T==Int32 ? "int32" : "float64","payload_bytes"=>N,"raw_offset_bytes"=>position(s.raw),"payload_offset_bytes"=>256,"guard_offset_bytes"=>position(s.guards),"guard_bytes"=>512,"stage"=>s.kind,"random"=>s.r,"step"=>s.step,"orbital"=>s.orbital)
  println(s.layout,json(row));write(s.raw,physical);write(s.guards,first);write(s.guards,last)
 end
 g=256÷sizeof(T);values=Float64.(host[g+1:g+b.count]);s.valid&=all(isfinite,values);values
end
function release!(s,b)
 paused(s)do
  if s.capture;s.released[b.id]=snapshot!(s,b,"release");else;snapshot!(s,b,"release");end
 end
 CUDA.unsafe_free!(b.full);delete!(s.live,b.id);nothing
end
mutable struct Wave{B}
 re::B;im::B;n::Int
end
function wave(s,n)
 re=allocate!(s,Float64,n);im=allocate!(s,Float64,n);dispatch!(Val(:zero),s,cld(n,256),Int32(n),re.data,im.data);Wave(re,im,Int(n))
end
function copywave(s,v)
 re=allocate!(s,Float64,v.n);im=allocate!(s,Float64,v.n);dispatch!(Val(:copy),s,cld(v.n,256),Int32(v.n),v.re.data,v.im.data,re.data,im.data);Wave(re,im,v.n)
end
function closewave(s,v)
 CUDA.device_synchronize();release!(s,v.re);release!(s,v.im)
end
function swapwave!(a,b)
 a.re,b.re=b.re,a.re;a.im,b.im=b.im,a.im;nothing
end
function copywave!(s,v,a)
 dispatch!(Val(:copy),s,cld(v.n,256),Int32(v.n),a.re.data,a.im.data,v.re.data,v.im.data)
end
function addwave!(s,v,a)
 dispatch!(Val(:add),s,cld(v.n,256),Int32(v.n),a.re.data,a.im.data,v.re.data,v.im.data)
end
struct Problem
 header::Vector{Int32};volume::Float64;emax::Float64;energy::Vector{Float64};time::Vector{Float64};locals::Vector{Int32};nn::Vector{Int32};nl::Vector{Int32};potential::Vector{Float64};hr::Vector{Float64};hi::Vector{Float64};dx::Vector{Float64};states::Vector{Tuple{Vector{Float64},Vector{Float64}}}
end
function problem(path)
 open(path)do io
  h=read!(io,Vector{Int32}(undef,12));n,maxn,ne,nt,M,R=h[1:6];volume=read(io,Float64);emax=read(io,Float64)
  rd(T,count)=read!(io,Vector{T}(undef,count));energy=rd(Float64,ne);time=rd(Float64,nt);locals=rd(Int32,h[12]);nn=rd(Int32,n);nl=rd(Int32,n*maxn);pot=rd(Float64,n);hr=rd(Float64,n*maxn);hi=rd(Float64,n*maxn);dx=rd(Float64,n*maxn);states=[(rd(Float64,n),rd(Float64,n))for _=1:R];@assert eof(io)
  Problem(h,volume,emax,energy,time,locals,nn,nl,pot,hr,hi,dx,states)
 end
end
struct Hamiltonian{BI,BF}
 n::Int;emax::Float64;nn::BI;nl::BI;potential::BF;hr::BF;hi::BF;dx::BF
end
function upload!(s,values)
 b=allocate!(s,eltype(values),length(values));copyto!(b.data,values);CUDA.device_synchronize();b
end
function Hamiltonian(s,p)
 n=Int(p.header[1]);maxn=Int(p.header[2]);col(v)=vec(permutedims(reshape(v,maxn,n)))
 nn=upload!(s,p.nn);nl=upload!(s,col(p.nl));pot=upload!(s,p.potential);hr=upload!(s,col(p.hr));hi=upload!(s,col(p.hi));dx=upload!(s,col(p.dx));H=Hamiltonian(n,p.emax,nn,nl,pot,hr,hi,dx)
 if s.capture
  push!(s.prefix,s.case_id);append!(s.prefix,p.header);append!(s.prefix,(p.volume,p.emax));append!(s.prefix,p.energy);append!(s.prefix,p.time);append!(s.prefix,p.locals)
  for b in (nn,nl,pot,hr,hi,dx);append!(s.prefix,snapshot!(s,b,"input_Hamiltonian"));end
 end;H
end
function closeH(s,H)
 for b in (H.nn,H.nl,H.potential,H.hr,H.hi,H.dx);release!(s,b);end
end
function initialize!(s,v,re,im)
 copyto!(v.re.data,re);copyto!(v.im.data,im);CUDA.device_synchronize()
 if s.capture
  paused(s)do
   append!(s.prefix,(s.r,s.orbital,v.n));a=snapshot!(s,v.re,"input_state");b=snapshot!(s,v.im,"input_state");@assert a==re&&b==im;append!(s.prefix,a);append!(s.prefix,b)
  end
 end
end
Hargs(H)=(Int32(H.n),H.emax,H.nn.data,H.nl.data,H.potential.data,H.hr.data,H.hi.data)
function applyH(s,H,input,out)
 dispatch!(Val(:H),s,cld(H.n,256),Hargs(H)...,input.re.data,input.im.data,out.re.data,out.im.data)
end
function current!(s,H,input,out)
 dispatch!(Val(:current),s,cld(H.n,256),Int32(H.n),H.nn.data,H.nl.data,H.hr.data,H.hi.data,H.dx.data,input.re.data,input.im.data,out.re.data,out.im.data)
end
function commutator!(s,H,input,out)
 dispatch!(Val(:commutator),s,cld(H.n,256),Int32(H.n),H.emax,H.nn.data,H.nl.data,H.hr.data,H.hi.data,H.dx.data,input.re.data,input.im.data,out.re.data,out.im.data)
end
function tag!(s,code,n)
 append!(s.output,(code,s.case_id,s.kind,s.r,s.step,s.orbital,n))
end
function hostmoments(s,v)
 re=Array(v.re.data);im=Array(v.im.data);CUDA.device_synchronize()
 if s.capture;paused(s)do;tag!(s,1,v.n);append!(s.output,re);append!(s.output,im);end;end;re,im
end
function moments!(s,p,H,left,right,out)
 n=H.n;M=Int(p.header[5]);blocks=cld(n,256);a=copywave(s,right);b=wave(s,n);c=wave(s,n);partial=wave(s,blocks*M)
 dispatch!(Val(:ip1),s,blocks,Int32(n),a.re.data,a.im.data,left.re.data,left.im.data,partial.re.data,partial.im.data,Int32(0));applyH(s,H,a,b)
 dispatch!(Val(:ip1),s,blocks,Int32(n),b.re.data,b.im.data,left.re.data,left.im.data,partial.re.data,partial.im.data,Int32(blocks))
 for m=2:M-1
  dispatch!(Val(:poly),s,blocks,Hargs(H)...,a.re.data,a.im.data,b.re.data,b.im.data,c.re.data,c.im.data)
  dispatch!(Val(:ip1),s,blocks,Int32(n),c.re.data,c.im.data,left.re.data,left.im.data,partial.re.data,partial.im.data,Int32(m*blocks));swapwave!(a,b);swapwave!(b,c)
 end
 dispatch!(Val(:ip2),s,M,Int32(n),partial.re.data,partial.im.data,out.re.data,out.im.data)
 for v in (partial,c,b,a);closewave(s,v);end;nothing
end
libsin(x)=ccall((:sin,"libm.so.6"),Cdouble,(Cdouble,),x)
libcos(x)=ccall((:cos,"libm.so.6"),Cdouble,(Cdouble,),x)
libtan(x)=ccall((:tan,"libm.so.6"),Cdouble,(Cdouble,),x)
jzero(x)=ccall((:j0,"libm.so.6"),Cdouble,(Cdouble,),x)
jone(x)=ccall((:j1,"libm.so.6"),Cdouble,(Cdouble,),x)
jn(m,x)=ccall((:jn,"libm.so.6"),Cdouble,(Cint,Cdouble),m,x)
function evolve!(s,H,v,scaled,direction)
 n=H.n;a=copywave(s,v);b=wave(s,n);c=wave(s,n);applyH(s,H,v,b)
 dispatch!(Val(:cheb01),s,cld(n,256),Int32(n),a.re.data,a.im.data,b.re.data,b.im.data,v.re.data,v.im.data,jzero(scaled),2.0*jone(scaled),Int32(direction))
 for m=2:999999
  bm=jn(m,scaled);if -1e-15<bm<1e-15;break;end;bm*=2.0;k=mod(m,4);label=k==0 ? 1 : k==2 ? 2 : ((k==1&&direction==1)||(k==3&&direction==-1)) ? 3 : 4
  dispatch!(Val(:cheb2),s,cld(n,256),Hargs(H)...,a.re.data,a.im.data,b.re.data,b.im.data,c.re.data,c.im.data,v.re.data,v.im.data,bm,Int32(label));swapwave!(a,b);swapwave!(b,c)
 end
 for x in(c,b,a);closewave(s,x);end
end
function evolvex!(s,H,v,scaled,direction)
 n=H.n;a=copywave(s,v);ax=wave(s,n);b=wave(s,n);bx=wave(s,n);c=wave(s,n);cx=wave(s,n);applyH(s,H,v,b);commutator!(s,H,v,bx)
 dispatch!(Val(:cheb1x),s,cld(n,256),Int32(n),bx.re.data,bx.im.data,v.re.data,v.im.data,2.0*jone(scaled))
 for m=2:1000000
  bm=jn(m,scaled);if -1e-15<bm<1e-15;break;end;bm*=2.0;k=mod(m,4);label=k==1 ? 3 : k==3 ? 4 : ((k==0&&direction==1)||(k==2&&direction==-1)) ? 1 : 2
  dispatch!(Val(:cheb2x),s,cld(n,256),Hargs(H)...,H.dx.data,a.re.data,a.im.data,ax.re.data,ax.im.data,b.re.data,b.im.data,bx.re.data,bx.im.data,c.re.data,c.im.data,cx.re.data,cx.im.data,v.re.data,v.im.data,bm,Int32(label));swapwave!(a,b);swapwave!(b,c);swapwave!(ax,bx);swapwave!(bx,cx)
 end
 for x in(cx,c,bx,b,ax,a);closewave(s,x);end
end
function density(p,re,im)
 M=Int(p.header[5]);pi=3.141592653589793
 for k=0:M-1
  factor=1.0/(M+1);damp=(1-k*factor)*libcos(k*pi*factor)+libsin(k*pi*factor)*factor/libtan(pi*factor);re[k+1]*=damp;im[k+1]*=damp
 end
 out=zeros(length(p.energy))
 for e=eachindex(p.energy)
  x=p.energy[e]/p.emax;c0=1.0;c1=x;temp=re[2]*c1
  for k=2:M-1;c2=2.0*x*c1-c0;c0=c1;c1=c2;temp+=re[k+1]*c2;end
  temp*=2.0;temp+=re[1];temp*=2.0/(pi*p.volume);temp/=sqrt(1.0-x*x);out[e]=temp/p.emax
 end;out
end
function scientific!(s,values)
 s.capture||return;paused(s)do;tag!(s,2,length(values));append!(s.output,values);end
end
function finalstate!(s,role,v)
 s.capture||return;paused(s)do;tag!(s,3,v.n);push!(s.output,role);append!(s.output,snapshot!(s,v.re,"final_state"));append!(s.output,snapshot!(s,v.im,"final_state"));end
end
function output_open(name)
 io=open(name,"a");println("Info: open ./"*name*" for output ");flush(stdout);io
end
function output_values(io,values)
 for x in values;@printf(io,"%.6g ",x);end;println(io);flush(io)
end
function DOS!(s,p,H,initial,local=false)
 out=wave(s,Int(p.header[5]));moments!(s,p,H,initial,initial,out);re,im=hostmoments(s,out);values=density(p,re,im);scientific!(s,values);io=output_open(local ? "ldos.out" : "dos.out");output_values(io,values);close(io);closewave(s,out)
end
function VAC0!(s,p,H,initial)
 out=wave(s,Int(p.header[5]));v=wave(s,H.n);current!(s,H,initial,v);moments!(s,p,H,v,v,out);re,im=hostmoments(s,out);values=density(p,re,im);scientific!(s,values);io=output_open("vac0.out");output_values(io,values);close(io);closewave(s,v);closewave(s,out)
end
function VAC!(s,p,H,initial)
 left=copywave(s,initial);copy=wave(s,H.n);right=copywave(s,initial);out=wave(s,Int(p.header[5]));current!(s,H,left,right);io=output_open("vac.out");nt=length(p.time)
 for step=0:nt-1
  s.step=step;println("- calculating VAC step $step");flush(stdout);current!(s,H,left,copy);moments!(s,p,H,right,copy,out);re,im=hostmoments(s,out);values=density(p,re,im);scientific!(s,values);output_values(io,values)
  if step<nt-1;scaled=p.time[step+1]*p.emax;evolve!(s,H,left,scaled,-1);evolve!(s,H,right,scaled,-1);end
 end
 finalstate!(s,10,left);finalstate!(s,11,right);close(io);for v in(out,right,copy,left);closewave(s,v);end
end
function MSD!(s,p,H,initial)
 state=copywave(s,initial);statex=copywave(s,initial);copy=wave(s,H.n);out=wave(s,Int(p.header[5]));scaled=p.time[1]*p.emax;evolve!(s,H,state,scaled,1);evolvex!(s,H,statex,scaled,1);io=output_open("msd.out");nt=length(p.time)
 for step=0:nt-1
  s.step=step;println("- calculating MSD step $step");flush(stdout);moments!(s,p,H,statex,statex,out);re,im=hostmoments(s,out);values=density(p,re,im);scientific!(s,values);output_values(io,values)
  if step<nt-1
   scaled=p.time[step+2]*p.emax;copywave!(s,copy,state);evolvex!(s,H,copy,scaled,1);evolve!(s,H,statex,scaled,1);addwave!(s,statex,copy);evolve!(s,H,state,scaled,1)
  end
 end
 finalstate!(s,12,state);finalstate!(s,13,statex);close(io);for v in(out,copy,statex,state);closewave(s,v);end
end
function spin!(s,p,H,initial)
 state=copywave(s,initial);sz=wave(s,H.n);out=wave(s,Int(p.header[5]));io=output_open("S.out");nt=length(p.time)
 for step=0:nt-1
  s.step=step;println("- calculating spin step $step");flush(stdout);dispatch!(Val(:sz),s,cld(H.n,256),Int32(H.n),state.re.data,state.im.data,sz.re.data,sz.im.data);moments!(s,p,H,state,sz,out);re,im=hostmoments(s,out);values=density(p,re,im);scientific!(s,values);output_values(io,values)
  if step<nt-1;evolve!(s,H,state,p.time[step+1]*p.emax,1);end
 end
 finalstate!(s,14,state);close(io);for v in(out,sz,state);closewave(s,v);end
end
function pipeline!(s,p)
 for name in("dos.out","vac0.out","vac.out","msd.out","S.out","ldos.out");isfile(name)&&rm(name);end
 H=Hamiltonian(s,p);initial=wave(s,H.n)
 for r=0:Int(p.header[6])-1
  s.r=r;s.kind=0;s.step=0;s.orbital=-1;initialize!(s,initial,p.states[r+1]...);timed(s,1)do;DOS!(s,p,H,initial);end
  if p.header[7]!=0;timed(s,2)do;VAC0!(s,p,H,initial);end;end
  if p.header[8]!=0;timed(s,3)do;VAC!(s,p,H,initial);end;end
  if p.header[9]!=0;timed(s,4)do;MSD!(s,p,H,initial);end;end
  if p.header[10]!=0;timed(s,5)do;spin!(s,p,H,initial);end;end
 end
 if p.header[11]!=0
  s.r=-1;timed(s,6)do
   for orbital in p.locals
    s.orbital=orbital;s.step=0;re=Vector{Float64}(undef,H.n);im=Vector{Float64}(undef,H.n);for i=1:H.n;re[i]=0.0;im[i]=0.0;end;re[orbital+1]=sqrt(Float64(H.n));initialize!(s,initial,re,im);DOS!(s,p,H,initial,true)
   end
  end
 end
 closewave(s,initial);closeH(s,H)
end
function begin_case!(s,folder,trial,case)
 @assert isempty(s.live);s.trial=trial;s.case_id=case;s.next_id=0;s.r=0;s.kind=0;s.step=0;s.orbital=-1;fill!(s.launches,0);fill!(s.timed_launches,0);empty!(s.released);s.capture||return
 base=joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_case$(case)");s.raw=open(base*".raw.bin","w");s.guards=open(base*".guards.u8","w");s.layout=open(base*".layout.jsonl","w");s.work=open(base*".work.json","w")
end
function end_case!(s)
 @assert isempty(s.live);s.capture||return
 for id in sort!(collect(keys(s.released)))
  v=s.released[id];append!(s.output,(4,s.case_id,id,length(v)));append!(s.output,v)
 end
 append!(s.output,(5,s.case_id,14));append!(s.output,s.launches);append!(s.output,s.timed_launches);println(s.work,json(Dict("case_id"=>s.case_id,"launches"=>s.launches,"timed_launches"=>s.timed_launches,"GPU_allocations"=>s.next_id)))
 close(s.raw);close(s.guards);close(s.layout);close(s.work)
end
function main()
 length(ARGS)>=2||error("fixture root and persistent output folder required");folder=ARGS[2];mkpath(folder);pilot=length(ARGS)==3&&ARGS[3]=="pilot";names=["diffusive","localized","small_lattice","general_tail","spin_pairs"];problems=[problem(joinpath(ARGS[1],name*"_fixture.bin"))for name in names];s=StudyState()
 for trial=0:(pilot ? 1 : 5)
  s.capture=trial>0;s.valid=true;s.seconds=0.0;empty!(s.prefix);empty!(s.output);started=time_ns()
  for case=0:4;begin_case!(s,folder,trial,case);pipeline!(s,problems[case+1]);end_case!(s);end
  trial==0&&continue;prefix=length(s.prefix);@assert prefix==769639;append!(s.prefix,s.output);s.valid&=all(isfinite,s.prefix);path="trial$(trial).bin";write(path,s.prefix)
  row=Dict("trial"=>trial,"time_seconds"=>s.seconds,"correct"=>s.valid,"checked_values"=>length(s.prefix),"max_abs_error"=>nothing,"max_rel_error"=>nothing,"input_prefix_values"=>prefix,"instrumented_driver_wall_seconds"=>(time_ns()-started)*1e-9,"timing_region"=>"sum_source_run_DOS_VAC0_VAC_MSD_spin_LDOS_functions_hostmath_alloc_IO_lifetimes_instrumented","jit_excluded"=>true,"internal_repeats"=>1,"output_artifact"=>Dict("path"=>path,"dtype"=>"float64","count"=>length(s.prefix),"byte_order"=>"little"));println("STUDY_TRIAL "*json(row));flush(stdout);s.valid||error("actualguard/finitecheckfailed")
 end
end
