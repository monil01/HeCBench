# Complete own Julia coarse-to-fine Horn–Schunck host pipeline and evidence lifetimes.
using Printf
json(x::AbstractString)="\""*replace(x,'\\'=>"\\\\",'\"'=>"\\\"")*"\""
json(x::Bool)=x ? "true" : "false"
json(x::Nothing)="null"
json(x::Number)=isfinite(x) ? string(x) : "null"
json(x::AbstractVector)="["*join(json.(x),",")*"]"
json(x::AbstractDict)="{"*join((json(string(k))*":"*json(v) for (k,v) in x),",")*"}"
mutable struct State
 capture::Bool;running::Bool;valid::Bool;next::Int;case_id::Int;seconds::Float64;mark::UInt64;checked::Int;max_abs::Float64;max_fraction::Float64;prefix::Vector{Float32};output::Vector{Float32};raw::IO;layout::IO;live::Dict{Int,Any};launches::Int
end
State()=State(false,false,true,0,0,0.,UInt64(0),0,0.,0.,Float32[],Float32[],devnull,devnull,Dict{Int,Any}(),0)
function paused(f,s)
 active=s.running
 if active;CUDA.device_synchronize();s.seconds+=(time_ns()-s.mark)*1e-9;s.running=false;end
 try;f();finally;if active;s.mark=time_ns();s.running=true;end;end
end
struct Buffer{F,D}
 id::Int;n::Int;full::F;data::D
end
function allocation(s,n)
 n=Int(n);full=CuArray{UInt8,1,CUDA.DeviceMemory}(undef,n*4+512);CUDA.device_synchronize()
 paused(s)do;fill!(full,UInt8(0xa7));CUDA.device_synchronize();end
 data=unsafe_wrap(CuArray{Float32,1,CUDA.DeviceMemory},reinterpret(CUDA.CuPtr{Float32},pointer(full)+256),n;own=false)
 b=Buffer(s.next,n,full,data);s.next+=1;s.live[b.id]=b;b
end
function release!(s,b)
 CUDA.device_synchronize()
 paused(s)do
  if s.capture
   raw=Array(b.full);s.valid&=all(==(0xa7),raw[1:256])&&all(==(0xa7),raw[end-255:end]);offset=position(s.raw);write(s.raw,UInt64(s.case_id),UInt64(b.id),UInt64(b.n*4),raw)
   println(s.layout,json(Dict("case"=>s.case_id,"id"=>b.id,"payload_bytes"=>b.n*4,"offset_bytes"=>offset,"actual_guard_bytes"=>512)))
  end
 end
 CUDA.unsafe_free!(b.data);CUDA.unsafe_free!(b.full);delete!(s.live,b.id);nothing
end
function readgpu(b)
 v=Array(b.data);CUDA.device_synchronize();v
end
function copyin!(b,x)
 copyto!(b.data,x);CUDA.device_synchronize();nothing
end
function active(v,w,h,stride)
 result=Vector{Float32}(undef,w*h);for y=0:h-1;copyto!(result,y*w+1,v,y*stride+1,w);end;result
end
function append_active!(s,b,w,h,stride)
 append!(s.output,active(readgpu(b),Int(w),Int(h),Int(stride)))
end
function compare!(s,a,r,bounds)
 @assert length(a)==length(r)==length(bounds)
 for i in eachindex(a)
  e=abs(Float64(a[i])-r[i]);s.valid&=isfinite(a[i])&&e<=bounds[i];s.max_abs=max(s.max_abs,e);if bounds[i]>0;s.max_fraction=max(s.max_fraction,e/bounds[i]);end;s.checked+=1
 end
end
function loadfixture(path)
 open(path)do io
  h=read!(io,Vector{UInt32}(undef,8));alpha=read(io,Float32);@assert h[1]==0x4f463537 && h[8]==2*h[4]*h[3];n=Int(h[4]*h[3]);a=read!(io,Vector{Float32}(undef,n));b=read!(io,Vector{Float32}(undef,n));@assert eof(io)
  (w=Int32(h[2]),h=Int32(h[3]),stride=Int32(h[4]),levels=Int(h[5]),warps=Int(h[6]),solver=Int(h[7]),alpha=alpha,a=a,b=b)
 end
end
function flow!(s,a)
 println("Computing optical flow on GPU...")
 levels=a.levels;p0=Vector{Any}(undef,levels);p1=similar(p0);pw=Vector{Int32}(undef,levels);ph=similar(pw);ps=similar(pw);n=length(a.a)
 tmp=allocation(s,n);du0=allocation(s,n);dv0=allocation(s,n);du1=allocation(s,n);dv1=allocation(s,n);gx=allocation(s,n);gy=allocation(s,n);gt=allocation(s,n);u=allocation(s,n);v=allocation(s,n);nu=allocation(s,n);nv=allocation(s,n)
 p0[levels]=allocation(s,n);p1[levels]=allocation(s,n);copyin!(p0[levels],a.a);copyin!(p1[levels],a.b)
 if s.capture
  paused(s)do
   append!(s.prefix,Float32[s.case_id,a.w,a.h,a.stride,a.alpha,levels,a.warps,a.solver]);append!(s.prefix,readgpu(p0[levels]));append!(s.prefix,readgpu(p1[levels]))
  end
 end
 pw[levels]=a.w;ph[levels]=a.h;ps[levels]=a.stride
 for l=levels:-1:2
  nw=div(pw[l],Int32(2));nh=div(ph[l],Int32(2));ns=cld(nw,Int32(256))*Int32(256);p0[l-1]=allocation(s,ns*nh);p1[l-1]=allocation(s,ns*nh)
  dispatch!(Val(:down),p0[l].data,pw[l],ph[l],ps[l],nw,nh,ns,p0[l-1].data);dispatch!(Val(:down),p1[l].data,pw[l],ph[l],ps[l],nw,nh,ns,p1[l-1].data);s.launches+=2;pw[l-1]=nw;ph[l-1]=nh;ps[l-1]=ns
 end
 fill!(u.data,0f0);fill!(v.data,0f0)
 for l=1:levels
  for warp=1:a.warps
   for b in(du0,dv0,du1,dv1);fill!(b.data,0f0);end
   dispatch!(Val(:warp),p1[l].data,pw[l],ph[l],ps[l],u.data,v.data,tmp.data);dispatch!(Val(:derivatives),p0[l].data,tmp.data,pw[l],ph[l],ps[l],gx.data,gy.data,gt.data);s.launches+=2
   for k=1:a.solver
    dispatch!(Val(:jacobi),du0.data,dv0.data,gx.data,gy.data,gt.data,pw[l],ph[l],ps[l],a.alpha,du1.data,dv1.data);du0,du1=du1,du0;dv0,dv1=dv1,dv0;s.launches+=1
   end
   dispatch!(Val(:add),u.data,du0.data,ph[l]*ps[l],u.data);dispatch!(Val(:add),v.data,dv0.data,ph[l]*ps[l],v.data);s.launches+=2
  end
  if l!=levels
   scalex=Float32(pw[l+1])/Float32(pw[l]);dispatch!(Val(:up),u.data,pw[l],ph[l],ps[l],pw[l+1],ph[l+1],ps[l+1],scalex,nu.data)
   scaley=Float32(ph[l+1])/Float32(ph[l]);dispatch!(Val(:up),v.data,pw[l],ph[l],ps[l],pw[l+1],ph[l+1],ps[l+1],scaley,nv.data);s.launches+=2;u,nu=nu,u;v,nv=nv,v
  end
 end
 hostu=readgpu(u);hostv=readgpu(v)
 for l=1:levels;release!(s,p0[l]);release!(s,p1[l]);end
 for b in(tmp,du0,dv0,du1,dv1,gx,gy,gt,nu,nv,u,v);release!(s,b);end
 hostu,hostv
end
function maincase!(s,root,name,folder,trial,case_id)
 a=loadfixture(joinpath(root,name*"_fixture.bin"));s.case_id=case_id;s.next=0
 if case_id==0;s.mark=time_ns();s.running=true;end
 u,v=flow!(s,a);CUDA.device_synchronize()
 if case_id==0;s.seconds+=(time_ns()-s.mark)*1e-9;s.running=false;end
 if trial==0;return;end
 open(joinpath(folder,"$(MODEL)_trial$(trial)_case$(case_id).host.bin"),"w")do io;write(io,u,v);end
 values=vcat(active(u,Int(a.w),Int(a.h),Int(a.stride)),active(v,Int(a.w),Int(a.h),Int(a.stride)))
 ref=open(joinpath(root,name*"_oracle.bin"))do io;h=read!(io,Vector{UInt32}(undef,2));@assert h==UInt32[a.w,a.h];r=read!(io,Vector{Float64}(undef,length(values)));@assert eof(io);r;end
 compare!(s,values,ref,0.05 .+ 0.001 .* abs.(ref));legacy=open(joinpath(root,name*"_legacyCPU.bin"))do io;h=read!(io,Vector{UInt32}(undef,2));@assert h==UInt32[a.w,a.h];r=read!(io,Vector{Float32}(undef,length(values)));@assert eof(io);r;end
 oldL1=sum(abs.(Float64.(values).-legacy))/(Int(a.w)*Int(a.h));append!(s.output,values)
 println("STUDY_SCIENCE "*json(Dict("case"=>case_id,"source_CPU_meanL1"=>oldL1,"source_CPU_threshold"=>.05,"boundary_policies_differ"=>true,"independent_max_abs"=>s.max_abs,"independent_max_bound_fraction"=>s.max_fraction)))
 if case_id==1;s.valid&=all(x->abs(x)<=.05f0,values);end
end
function primitivecase!(s,root,trial)
 s.case_id=2;s.next=0;ptr=Any[]
 open(joinpath(root,"primitive_fixture.bin"))do io
  @assert read(io,UInt32)==0x4f465052&&read(io,UInt32)==9
  for id=0:8
   h=read!(io,Vector{UInt32}(undef,4));@assert h[1]==id;w,height,stride=Int.(h[2:4]);n=height*stride;host=read!(io,Vector{Float32}(undef,n));b=allocation(s,n);copyin!(b,host);push!(ptr,b)
   if trial>0;append!(s.prefix,Float32[id,w,height,stride,n]);append!(s.prefix,readgpu(b));end
  end
  @assert eof(io)
 end
 if trial>0;push!(s.prefix,.2f0);end
 o=allocation(s,256*64);x=allocation(s,256*32);y=allocation(s,256*32);z=allocation(s,256*32);output_start=length(s.output)
 dispatch!(Val(:down),ptr[1].data,Int32(64),Int32(32),Int32(256),Int32(32),Int32(16),Int32(256),o.data);s.launches+=1;if trial>0;append_active!(s,o,32,16,256);end
 dispatch!(Val(:up),ptr[1].data,Int32(64),Int32(32),Int32(256),Int32(128),Int32(64),Int32(256),2f0,o.data);s.launches+=1;if trial>0;append_active!(s,o,128,64,256);end
 dispatch!(Val(:warp),ptr[2].data,Int32(64),Int32(32),Int32(256),ptr[3].data,ptr[4].data,o.data);s.launches+=1;if trial>0;append_active!(s,o,64,32,256);end
 dispatch!(Val(:derivatives),ptr[1].data,ptr[2].data,Int32(64),Int32(32),Int32(256),x.data,y.data,z.data);s.launches+=1;if trial>0;for b in(x,y,z);append_active!(s,b,64,32,256);end;end
 dispatch!(Val(:add),ptr[1].data,ptr[2].data,Int32(256*32),o.data);s.launches+=1;CUDA.device_synchronize();if trial>0;v=Vector{Float32}(undef,256*32);copyto!(v,1,o.data,1,length(v));CUDA.device_synchronize();append!(s.output,v);end
 du=allocation(s,256*13);dv=allocation(s,256*13)
 dispatch!(Val(:jacobi),ptr[5].data,ptr[6].data,ptr[7].data,ptr[8].data,ptr[9].data,Int32(65),Int32(13),Int32(256),.2f0,du.data,dv.data);s.launches+=1;CUDA.device_synchronize();if trial>0;append_active!(s,du,65,13,256);append_active!(s,dv,65,13,256);end
 fill!(ptr[5].data,0f0);fill!(ptr[6].data,0f0);CUDA.device_synchronize();if trial>0;for b in(ptr[5],ptr[6]);append!(s.prefix,readgpu(b));end;end
 oldu,oldv,newu,newv=ptr[5],ptr[6],du,dv
 for k=1:500
  dispatch!(Val(:jacobi),oldu.data,oldv.data,ptr[7].data,ptr[8].data,ptr[9].data,Int32(65),Int32(13),Int32(256),.2f0,newu.data,newv.data);oldu,newu=newu,oldu;oldv,newv=newv,oldv;s.launches+=1
 end
 CUDA.device_synchronize()
 if trial>0
  append_active!(s,oldu,65,13,256);append_active!(s,oldv,65,13,256)
  open(joinpath(root,"primitive_oracle.bin"))do io
   @assert read(io,UInt32)==0x4f464f52&&read(io,UInt32)==11&&read(io,UInt32)==28468;sizes=read!(io,Vector{UInt32}(undef,11));r=read!(io,Vector{Float64}(undef,28468));b=read!(io,Vector{Float64}(undef,28468));@assert eof(io);compare!(s,s.output[output_start+1:end],r,b)
  end
 end
 for b in ptr;release!(s,b);end;for b in(o,x,y,z,du,dv);release!(s,b);end
end
function main()
 @assert length(ARGS)==2;root,folder=ARGS;mkpath(folder);s=State()
 for trial=0:5
  s.capture=trial>0;s.running=false;s.valid=true;s.seconds=s.max_abs=s.max_fraction=0.;s.checked=0;s.launches=0;empty!(s.prefix);empty!(s.output)
  if trial>0;s.raw=open(joinpath(folder,"$(MODEL)_trial$(trial).physical.bin"),"w");s.layout=open(joinpath(folder,"$(MODEL)_trial$(trial).layout.jsonl"),"w");end
  maincase!(s,root,"original_pair",folder,trial,0);maincase!(s,root,"identity_partial",folder,trial,1);primitivecase!(s,root,trial);@assert isempty(s.live)
  if trial==0;continue;end
  close(s.raw);close(s.layout);prefix=length(s.prefix);append!(s.prefix,s.output);s.valid&=all(isfinite,s.prefix)&&s.launches==12626;path="trial$(trial).bin";write(path,s.prefix)
  println("STUDY_TRIAL "*json(Dict("trial"=>trial,"time_seconds"=>s.seconds,"correct"=>s.valid,"checked_values"=>length(s.prefix),"max_abs_error"=>s.max_abs,"max_rel_error"=>nothing,"independent_max_bound_fraction"=>s.max_fraction,"independent_checked_values"=>s.checked,"input_prefix_values"=>prefix,"authored_kernel_launches"=>s.launches,"timing_region"=>"fullsource_ComputeFlowCUDA640x480_5levels3warps500Jacobi_alloc_copies_free_observer_split","jit_excluded"=>true,"internal_repeats"=>1,"output_artifact"=>Dict("path"=>path,"dtype"=>"float32","count"=>length(s.prefix),"byte_order"=>"little"))));flush(stdout);s.valid||error("fullflow/primitive/memoryguard/workcountfailure")
 end
end
