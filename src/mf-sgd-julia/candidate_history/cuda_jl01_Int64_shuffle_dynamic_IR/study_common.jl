# Own Julia cuMF_SGD host pipeline and source-aligned observation lifetimes.
using Printf
struct MFNode
 u::Int32;v::Int32;rate::Float32
end
@assert sizeof(MFNode)==12 && fieldoffset(MFNode,2)==4 && fieldoffset(MFNode,3)==8
json(x::AbstractString)="\""*replace(x,'\\'=>"\\\\",'\"'=>"\\\"")*"\""
json(x::Bool)=x ? "true" : "false"
json(x::Nothing)="null"
json(x::Number)=isfinite(x) ? string(x) : "null"
json(x::AbstractVector)="["*join(json.(x),",")*"]"
json(x::AbstractDict)="{"*join((json(string(k))*":"*json(v) for (k,v) in x),",")*"}"
struct Inputs
 m::Int;n::Int;k::Int;nr::Int;nnz::Int;nt::Int;parameters::Vector{Float32};pm::Vector{Int32};qm::Vector{Int32};ip::Vector{Int32};iq::Vector{Int32};R::Vector{MFNode};test::Vector{MFNode};original::Vector{MFNode};P::Vector{Float16};Q::Vector{Float16};rates::Vector{Float32}
end
function inputs(path,original_path)
 open(path) do io
  @assert read(io,UInt32)==0x4d465331 && read(io,UInt32)==1
  h=read!(io,Vector{Int32}(undef,4));m,n,k,nr=Int.(h);nnz=Int(read(io,Int64));nt=Int(read(io,Int64));rd(T,N)=read!(io,Vector{T}(undef,N))
  parameters=rd(Float32,9);pm=rd(Int32,m);qm=rd(Int32,n);ip=rd(Int32,m);iq=rd(Int32,n);R=rd(MFNode,nnz);test=rd(MFNode,nt);P=rd(Float16,m*k);Q=rd(Float16,n*k);rates=rd(Float32,nr);@assert eof(io)
  original=open(f->read!(f,Vector{MFNode}(undef,nnz)),original_path)
  Inputs(m,n,k,nr,nnz,nt,parameters,pm,qm,ip,iq,R,test,original,P,Q,rates)
 end
end
mutable struct StudyState
 capture::Bool;roi::Bool;pause_depth::Int;mark::UInt64;seconds::Float64;valid::Bool;case_id::Int;workers::Int;epochs::Int;m::Int;n::Int;nnz::Int;next_id::Int
 live::Dict{Int,Any};released::Dict{Int,Vector{UInt8}};prefix::Vector{Float64};launches::Vector{Int};raw::IO;guards::IO;layout::IO
end
StudyState()=StudyState(true,false,0,UInt64(0),0.,true,0,0,30,0,0,0,0,Dict{Int,Any}(),Dict{Int,Vector{UInt8}}(),Float64[],zeros(Int,4),devnull,devnull,devnull)
function pause!(s)
 if s.roi;s.pause_depth+=1;if s.pause_depth==1;s.seconds+=(time_ns()-s.mark)*1e-9;end;end
end
function resume!(s)
 if s.roi;s.pause_depth-=1;if s.pause_depth==0;s.mark=time_ns();end;end
end
function paused(f,s)
 pause!(s);try;f();finally;resume!(s);end
end
function start!(s)
 @assert !s.roi && s.pause_depth==0;s.roi=true;s.mark=time_ns()
end
function stop!(s)
 CUDA.device_synchronize();@assert s.pause_depth==0;s.seconds+=(time_ns()-s.mark)*1e-9;s.roi=false
end
struct GpuBuffer{T,F,D}
 id::Int;full::F;data::D;n::Int
end
function gpu_alloc(s,::Type{T},n;tracked=true) where T
 n=Int(n);full=CuArray{UInt8,1,CUDA.DeviceMemory}(undef,n*sizeof(T)+512);CUDA.device_synchronize()
 paused(s) do
  copyto!(full,fill(UInt8(165),n*sizeof(T)+512));CUDA.device_synchronize()
 end
 # Nonowning wrapper avoids hidden retained view leases at explicit source frees.
 data=unsafe_wrap(CuArray{T,1,CUDA.DeviceMemory},reinterpret(CUDA.CuPtr{T},pointer(full)+256),n;own=false)
 id=tracked ? s.next_id : 1000;if tracked;s.next_id+=1;end;b=GpuBuffer{T,typeof(full),typeof(data)}(id,full,data,n);if tracked;s.live[id]=b;end;b
end
function close_gpu!(b)
 CUDA.unsafe_free!(b.data);CUDA.unsafe_free!(b.full);nothing
end
function snapshot!(s,b,event)
 bytes=Array(b.full);N=length(bytes)-512;first=bytes[1:256];last=bytes[end-255:end];s.valid&=all(==(165),first)&&all(==(165),last)
 if s.capture
  row=Dict("case"=>s.case_id,"id"=>b.id,"event"=>event,"payload_bytes"=>N,"raw_offset_bytes"=>position(s.raw),"guard_offset_bytes"=>position(s.guards),"guard_bytes"=>512)
  println(s.layout,json(row));write(s.raw,bytes);write(s.guards,first);write(s.guards,last)
 end;bytes[257:256+N]
end
function gpu_release(s,b)
 CUDA.device_synchronize();paused(s) do;s.released[b.id]=snapshot!(s,b,"release");end;close_gpu!(b);delete!(s.live,b.id);nothing
end
struct HostBuffer{T}
 full::Vector{UInt8};data::Vector{T}
end
function host_buffer(::Type{T},n) where T
 full=fill(UInt8(165),n*sizeof(T)+512);CUDA.pin(full);data=unsafe_wrap(Vector{T},Ptr{T}(pointer(full)+256),n;own=false);HostBuffer{T}(full,data)
end
host_guards(h)=all(==(165),h.full[1:256])&&all(==(165),h.full[end-255:end])
save_host(h,path)=write(path,h.full)
close_host(h)=finalize(h.full)
function copy_in!(b,h::HostBuffer)
 GC.@preserve b h begin;copyto!(b.data,h.data);CUDA.device_synchronize();end
end
function copy_in!(b,v::Vector)
 GC.@preserve b v begin;copyto!(b.data,v);CUDA.device_synchronize();end
end
function copy_out!(h,b)
 GC.@preserve h b begin;copyto!(h.data,b.data);CUDA.device_synchronize();end
end
function open_case!(s,base,id,w,m,n,nnz,epochs)
 @assert isempty(s.live);s.case_id=id;s.workers=w;s.m=m;s.n=n;s.nnz=nnz;s.epochs=epochs;s.next_id=0;empty!(s.released);fill!(s.launches,0)
 s.raw=open(base*".raw.bin","w");s.guards=open(base*".guards.u8","w");s.layout=open(base*".layout.jsonl","w")
end
function close_case!(s)
 @assert isempty(s.live);close(s.raw);close(s.guards);close(s.layout)
end
function before_training!(s)
 CUDA.device_synchronize();paused(s) do
  append!(s.prefix,(s.case_id,s.workers,s.epochs,s.m,s.n,s.nnz,128))
  for id=0:4
   bytes=snapshot!(s,s.live[id],"actual_before_training");append!(s.prefix,(id,length(bytes)))
   if id==0;append!(s.prefix,Float64.(reinterpret(UInt32,bytes)))
   elseif id==1;append!(s.prefix,Float64.(reinterpret(Float32,bytes)[1:s.epochs+4]))
   elseif id==2;for node in reinterpret(MFNode,bytes);append!(s.prefix,(node.u,node.v,node.rate));end
   else;append!(s.prefix,Float64.(reinterpret(UInt16,bytes)));end
  end
 end
end
function before_feature!(s,state_size,array_size,k,scale,grid)
 CUDA.device_synchronize();paused(s) do
  append!(s.prefix,(s.case_id,state_size,array_size,k,scale,grid));append!(s.prefix,Float64.(reinterpret(UInt32,snapshot!(s,s.live[1],"actual_feature_state_input"))))
 end
end
function common_prefix!(s,x)
 append!(s.prefix,(UInt32(0x4d465331),1,x.m,x.n,x.k,x.nnz,x.nt));append!(s.prefix,Float64.(x.parameters))
 for v in(x.pm,x.qm,x.ip,x.iq);append!(s.prefix,Float64.(v));end
 for node in x.original;append!(s.prefix,(node.u,node.v,node.rate));end
 for node in x.test;append!(s.prefix,(node.u,node.v,node.rate));end
 append!(s.prefix,(257,128,12,Float64(.0125f0),Float64(.05f0),Float64(.075f0)))
end
function validate_rng(trace,workers,epochs,updates,vector_size,R,m,n,actual_bytes)
 states=UInt32[lcg_next(UInt32(5551212)⊻UInt32(i)) for i=0:workers-1];counts=zeros(Int32,length(R));users=zeros(Int32,m);items=zeros(Int32,n);valid=length(trace)==workers*epochs*updates
 for epoch=0:epochs-1,group=0:updates-1,worker=0:workers-1
  states[worker+1]=lcg_next(states[worker+1]);start=mod(trunc(Int64,(Float32(states[worker+1])/2147483648.0f0)*Float32(length(R))),length(R));valid&=trace[(epoch*updates+group)*workers+worker+1]==start
  for j=0:vector_size-1;at=mod(start+j,length(R))+1;counts[at]+=1;users[R[at].u+1]+=1;items[R[at].v+1]+=1;end
 end
 valid&=reinterpret(UInt32,actual_bytes)==states;(valid,states,counts,users,items)
end
function append_rng!(out,rng,overall)
 valid,states,counts,users,items=rng;append!(out,(overall ? 1 : 0,valid ? 1 : 0));for v in(states,counts,users,items);append!(out,Float64.(v));end
end
libpow(x,y)=ccall((:pow,"libm.so.6"),Cdouble,(Cdouble,Cdouble),x,y)
function source_rates(x)
 out=collect(reinterpret(Float32,fill(UInt32(0xa5a5a5a5),1024)))
 for i=0:33;out[i+1]=Float32(x.parameters[5]/(1+x.parameters[6]*libpow(Float64(i),1.5))+x.parameters[7]);end
 @assert out[1:34]==x.rates;out
end
function predictions(P,Q,R,k)
 out=Vector{Float64}(undef,length(R))
 for i in eachindex(R)
  node=R[i];value=0.0
  for j=1:k;value+=Float64(P[Int(node.u)*k+j])*Float64(Q[Int(node.v)*k+j]);end;out[i]=value
 end;out
end
squared_loss(pred,R)=sum((pred[i]-Float64(R[i].rate))^2 for i in eachindex(R))
function write_science(base,x,workers,halfP,halfQ,beforeP,beforeQ,finalP,finalQ,pred,heldout)
 open(base*".science.bin","w") do io
  write(io,Int32[0x4d464d31,x.m,x.n,x.k,workers,30,x.nnz,x.nt]);write(io,x.parameters)
  for v in(halfP,halfQ,beforeP,beforeQ,finalP,finalQ,pred,heldout);write(io,v);end
 end
end
function postprocess!(P,Q,x)
 println("scale model ...");factor=Float32(sqrt(Float64(x.parameters[1])));started=time_ns()
 for i in eachindex(P);P[i]*=factor;end;for i in eachindex(Q);Q[i]*=factor;end
 @printf("time elapsed:%.8fs\n\n\n",(time_ns()-started)*1e-9);println("shuffle model ...");started=time_ns();oldP=copy(P);oldQ=copy(Q)
 for i=0:x.m-1,j=1:x.k;P[i*x.k+j]=oldP[Int(x.pm[i+1])*x.k+j];end
 for i=0:x.n-1,j=1:x.k;Q[i*x.k+j]=oldQ[Int(x.qm[i+1])*x.k+j];end
 @printf("time elapsed:%.8fs\n\n\n",(time_ns()-started)*1e-9)
end
function transform_host!(s,halfhost,floathost,seg,k)
 input=gpu_alloc(s,Float16,seg*k);output=gpu_alloc(s,Float32,seg*k);copy_in!(input,halfhost)
 blocks=min(cld(seg*k,256),192);dispatch!(Val(:transform),s,blocks,256,Int32(blocks*256),input.data,output.data,Int64(seg*k));copy_out!(floathost,output)
 gpu_release(s,input);gpu_release(s,output)
end
function source_sgd_update!(s,x,workers,hR,hP,hQ,fP,fQ,trace)
 println("sgd_update_k128 ...");native_diagnostic_start=time_ns()
 state=gpu_alloc(s,UInt32,workers);dispatch!(Val(:rng_sized),s,cld(workers,256),256,UInt32(5551212),state.data,Int32(workers))
 rate_values=source_rates(x);rates=gpu_alloc(s,Float32,1024);copy_in!(rates,rate_values)
 R=gpu_alloc(s,MFNode,x.nnz);p=gpu_alloc(s,Float16,length(x.P));q=gpu_alloc(s,Float16,length(x.Q));copy_in!(R,hR);copy_in!(p,hP);copy_in!(q,hQ)
 updates=cld(x.nnz,workers*128);before_training!(s)
 dispatch!(Val(:sgd),s,div(workers,4),128,R.data,Int64(x.nnz),p.data,q.data,state.data,rates.data,Int64(x.m),Int64(x.n),Int32(128),Int32(30),Int32(0),Int32(updates),Int32(updates),Int32(128),x.parameters[8],x.parameters[9],Int32(1),Int32(1),Int32(0),Int32(0),trace.data)
 copy_out!(hP,p);copy_out!(hQ,q);gpu_release(s,p);gpu_release(s,q);gpu_release(s,R)
 CUDA.device_synchronize();transform_host!(s,hP,fP,x.m,128);transform_host!(s,hQ,fQ,x.n,128);gpu_release(s,rates);gpu_release(s,state)
 @printf("time elapsed:%.8fs\n\n\n",(time_ns()-native_diagnostic_start)*1e-9)
end
function training_case!(s,x,workers,case,base,out)
 hR=host_buffer(MFNode,x.nnz);copyto!(hR.data,x.R);hP=host_buffer(Float16,length(x.P));hQ=host_buffer(Float16,length(x.Q));copyto!(hP.data,x.P);copyto!(hQ.data,x.Q);fP=host_buffer(Float32,length(x.P));fQ=host_buffer(Float32,length(x.Q));updates=cld(x.nnz,workers*128)
 trace=gpu_alloc(s,Int64,workers*30*updates;tracked=false);open_case!(s,base,case,workers,x.m,x.n,x.nnz,30);start!(s);source_sgd_update!(s,x,workers,hR,hP,hQ,fP,fQ,trace);stop!(s)
 actual_trace=collect(reinterpret(Int64,snapshot!(s,trace,"actual_selector_trace")));rng=validate_rng(actual_trace,workers,30,updates,128,x.R,x.m,x.n,s.released[0]);halfP=copy(hP.data);halfQ=copy(hQ.data);beforeP=copy(fP.data);beforeQ=copy(fQ.data)
 converted=reinterpret(UInt32,beforeP)==reinterpret(UInt32,Float32.(halfP))&&reinterpret(UInt32,beforeQ)==reinterpret(UInt32,Float32.(halfQ));postprocess!(fP.data,fQ.data,x);finalP=copy(fP.data);finalQ=copy(fQ.data);factor=Float32(sqrt(Float64(x.parameters[1])));initialP=Vector{Float32}(undef,length(x.P));initialQ=Vector{Float32}(undef,length(x.Q))
 for i=0:x.m-1,j=1:x.k;initialP[i*x.k+j]=Float32(x.P[Int(x.pm[i+1])*x.k+j])*factor;end
 for i=0:x.n-1,j=1:x.k;initialQ[i*x.k+j]=Float32(x.Q[Int(x.qm[i+1])*x.k+j])*factor;end
 pred=predictions(finalP,finalQ,x.original,x.k);heldout=predictions(finalP,finalQ,x.test,x.k);initial_pred=predictions(initialP,initialQ,x.original,x.k);loss=squared_loss(pred,x.original);initial_loss=squared_loss(initial_pred,x.original);test_rmse=sqrt(squared_loss(heldout,x.test)/x.nt)
 finite=all(isfinite,finalP)&&all(isfinite,finalQ)&&all(isfinite,pred)&&all(isfinite,heldout);changed=reinterpret(UInt16,halfP)!=reinterpret(UInt16,x.P)||reinterpret(UInt16,halfQ)!=reinterpret(UInt16,x.Q);hostguards=all(host_guards,(hR,hP,hQ,fP,fQ));valid=finite&&changed&&converted&&hostguards&&s.valid&&rng[1]&&loss<initial_loss&&test_rmse<=3.0
 append!(out,(case,valid ? 1 : 0,finite ? 1 : 0,changed ? 1 : 0,converted ? 1 : 0,hostguards ? 1 : 0,loss<initial_loss ? 1 : 0,test_rmse<=3 ? 1 : 0));append!(out,Float64.(actual_trace));append_rng!(out,rng,valid);append!(out,Float64.(s.launches))
 write_science(base,x,workers,halfP,halfQ,beforeP,beforeQ,finalP,finalQ,pred,heldout)
 for (h,name) in zip((hR,hP,hQ,fP,fQ),("R","halfP","halfQ","floatP","floatQ"));save_host(h,base*".host_"*name*".bin");end
 checks=Dict("case"=>case,"workers"=>workers,"epochs"=>30,"updates_per_worker_epoch"=>updates,"source_selector_draws"=>length(actual_trace),"rating_updates"=>length(actual_trace)*128,"property_pass"=>valid,"finite"=>finite,"source_half_to_float_exact"=>converted,"rng_exact"=>rng[1],"host_guards"=>hostguards,"initial_train_squared_error"=>initial_loss,"final_train_squared_error"=>loss,"heldout_dot_only_RMSE"=>test_rmse,"kernel_launches"=>copy(s.launches));write(base*".checks.json",json(checks)*"\n")
 close_case!(s);close_gpu!(trace);for h in(hR,hP,hQ,fP,fQ);close_host(h);end;valid
end
function source_init_feature!(s,output,seg,k)
 scale=Float32(sqrt(1.0/k));gpu=gpu_alloc(s,Float16,seg*k);state_size=(div(seg,256)+1)*256;println("state_size (a multiple of 256):",state_size);state=gpu_alloc(s,UInt32,state_size)
 dispatch!(Val(:rng_unsized),s,div(state_size,256),256,UInt32(5551212),state.data);blocks=cld(seg*k,256);println("\tnumber of thread blocks:",blocks);println("\tarraysize:",seg*k);println("grid:0")
 before_feature!(s,state_size,seg*k,k,scale,0);dispatch!(Val(:features),s,blocks,256,Int32(blocks*256),state.data,Int32(state_size),gpu.data,Int64(seg*k),Int64(k),scale);copy_out!(output,gpu);gpu_release(s,state);gpu_release(s,gpu)
end
function feature_case!(s,seg,case,base,out)
 output=host_buffer(Float16,seg*128);open_case!(s,base,case,0,seg,128,0,0);source_init_feature!(s,output,seg,128);bits=collect(reinterpret(UInt16,output.data));states=collect(reinterpret(UInt32,s.released[1]));valid=host_guards(output)&&s.valid;scale=Float32(sqrt(1.0/128))
 for i=0:255
  seed=lcg_next(UInt32(5551212)⊻UInt32(i))
  if i<seg*128;seed=lcg_next(seed);value=Float16((Float32(seed)/2147483648.0f0)*scale);valid&=bits[i+1]==reinterpret(UInt16,value);end
  valid&=states[i+1]==seed
 end
 valid&=s.released[0]==collect(reinterpret(UInt8,bits));append!(out,(case,valid ? 1 : 0,seg*128));append!(out,Float64.(bits));append!(out,Float64.(states));append!(out,Float64.(s.launches));save_host(output,base*".host_features.bin");close_case!(s);close_host(output);valid
end
function operator_case!(s,oracle,unit_fixture,base,out)
 n=257;k=128;workers=12
 P,Q,coefficients=open(unit_fixture) do io
  @assert read!(io,Vector{UInt32}(undef,3))==UInt32[257,128,12];c=read!(io,Vector{Float32}(undef,3));P=read!(io,Vector{Float16}(undef,n*k));Q=read!(io,Vector{Float16}(undef,n*k));@assert eof(io);(P,Q,c)
 end
 lr,lp,lq=coefficients;R=MFNode[MFNode(Int32(i),Int32(i),Float32(mod(i,7)-3)*.25f0) for i=0:n-1];open_case!(s,base,2,workers,n,n,n,1)
 state=gpu_alloc(s,UInt32,workers);rates=gpu_alloc(s,Float32,1024);dR=gpu_alloc(s,MFNode,n);dP=gpu_alloc(s,Float16,length(P));dQ=gpu_alloc(s,Float16,length(Q));rate_values=zeros(Float32,1024);rate_values[1]=lr
 copy_in!(rates,rate_values);copy_in!(dR,R);copy_in!(dP,P);copy_in!(dQ,Q);dispatch!(Val(:rng_sized),s,1,256,UInt32(5551212),state.data,Int32(workers));CUDA.device_synchronize();trace=gpu_alloc(s,Int64,workers;tracked=false);before_training!(s)
 dispatch!(Val(:sgd),s,3,128,dR.data,Int64(n),dP.data,dQ.data,state.data,rates.data,Int64(n),Int64(n),Int32(k),Int32(1),Int32(0),Int32(1),Int32(1),Int32(1),lp,lq,Int32(1),Int32(1),Int32(0),Int32(0),trace.data);CUDA.device_synchronize()
 finalP=Array(dP.data);finalQ=Array(dQ.data);actual_trace=collect(reinterpret(Int64,snapshot!(s,trace,"actual_selector_trace")));for b in(dP,dQ,dR,rates,state);gpu_release(s,b);end
 rng=validate_rng(actual_trace,workers,1,1,1,R,n,n,s.released[0]);valid=s.valid&&rng[1];max_error=0.;max_fraction=0.;selected=Int32[]
 open(oracle) do io
  @assert read!(io,Vector{UInt32}(undef,5))==UInt32[0x4d464f31,257,128,12,3072];append!(selected,read!(io,Vector{Int32}(undef,12)))
  for z=1:3072
   which=read(io,Int32);row=read(io,Int32);j=read(io,Int32);ideal=read(io,Float64);bound=read(io,Float64);got=Float64((which==0 ? finalP : finalQ)[Int(row)*k+Int(j)+1]);error=abs(got-ideal);valid&=isfinite(got)&&error<=bound;max_error=max(max_error,error);max_fraction=max(max_fraction,error/bound)
  end;@assert eof(io)
 end
 for i=0:n-1
  if !(i in selected)
   a=i*k+1;b=(i+1)*k;valid&=reinterpret(UInt16,finalP[a:b])==reinterpret(UInt16,P[a:b])&&reinterpret(UInt16,finalQ[a:b])==reinterpret(UInt16,Q[a:b])
  end
 end
 append!(out,(2,valid ? 1 : 0,3072));append!(out,Float64.(actual_trace));append_rng!(out,rng,valid);append!(out,Float64.(s.launches));open(base*".operator.bin","w") do io;write(io,finalP);write(io,finalQ);end
 write(base*".checks.json",json(Dict("property_pass"=>valid,"updated_components"=>3072,"unchanged_components"=>62720,"max_ideal_absolute_error"=>max_error,"max_derived_bound_fraction"=>max_fraction,"all12independentwarps"=>true))*"\n");close_case!(s);close_gpu!(trace);valid
end
function main()
 length(ARGS)>=4||error("fixture originaltrain oracle persistentroot [pilot]");x=inputs(ARGS[1],ARGS[2]);folder=ARGS[4];mkpath(folder);oracle=ARGS[3];unit_fixture=joinpath(dirname(oracle),"operator_fixture_v2.bin");pilot=length(ARGS)>=5&&ARGS[5]=="pilot";s=StudyState()
 println("STUDY_ENV ",json(Dict("model"=>MODEL,"julia"=>string(VERSION),"CUDA"=>string(Base.pkgversion(CUDA)),"gpu_parent_allocation"=>"explicit_CUDA_DeviceMemory_CuArray_pool","guarded_views"=>"nonowning_unsafe_wrap_parent_kept_in_live_buffer_map_explicit_free_both_wrappers")))
 for trial=0:(pilot ? 1 : 5)
  s.capture=true;s.valid=true;s.seconds=0.;empty!(s.prefix);common_prefix!(s,x);out=Float64[];valid=true;base(c)=joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_case$(c)")
  valid&=feature_case!(s,1,3,base(3),out);valid&=feature_case!(s,2,4,base(4),out);valid&=operator_case!(s,oracle,unit_fixture,base(2),out);valid&=training_case!(s,x,12,0,base(0),out);valid&=training_case!(s,x,128,1,base(1),out)
  if trial==0
   if !valid;write("warmup_failure.bin",vcat(s.prefix,out));error("warmupindependent/guard/propertycheckfailed");end;continue
  end
  prefix=length(s.prefix);@assert prefix==1524635;append!(s.prefix,out);@assert length(s.prefix)==1730047;path="trial$(trial).bin";write(path,s.prefix)
  row=Dict("trial"=>trial,"time_seconds"=>s.seconds,"correct"=>valid,"checked_values"=>length(s.prefix),"input_prefix_values"=>prefix,"max_abs_error"=>nothing,"max_rel_error"=>nothing,"timing_region"=>"sum_two_complete_source_onepartition_SGDupdatefunctions_30epochs_workers12and128_instrumented","jit_excluded"=>true,"internal_repeats"=>1,"process_id"=>getpid(),"auxiliary_raw_root"=>folder,"output_artifact"=>Dict("path"=>path,"dtype"=>"float64","count"=>length(s.prefix),"byte_order"=>"little"));println("STUDY_TRIAL "*json(row));flush(stdout);valid||error("property/guard/operatorcheckfailed")
 end
end
