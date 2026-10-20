# Own Julia complete source CNN host pipeline, same stage evidence order.
using Printf
json(x::AbstractString)="\""*replace(x,'\\'=>"\\\\",'\"'=>"\\\"")*"\""
json(x::Bool)=x ? "true" : "false"
json(x::Nothing)="null"
json(x::Number)=isfinite(x) ? string(x) : "null"
json(x::AbstractVector)="["*join(json.(x),",")*"]"
json(x::AbstractDict)="{"*join((json(string(k))*":"*json(v) for(k,v) in x),",")*"}"
mutable struct State
 capture::Bool;running::Bool;pending::Bool;step::Int;label::Int;oldstep::Int;oldlabel::Int;seconds::Float64;mark::UInt64;valid::Bool;prefix::Vector{Float64};output::Vector{Float64};science::IO;launches::Int;memsets::Int
end
State()=State(false,false,false,0,0,0,0,0.,UInt64(0),true,Float64[],Float64[],devnull,0,0)
function paused(f,s)
 active=s.running
 if active;CUDA.device_synchronize();s.seconds+=(time_ns()-s.mark)*1e-9;s.running=false;end
 try;f();finally;if active;s.mark=time_ns();s.running=true;end;end
end
struct Buffer{F,D}
 full::F;data::D;n::Int;id::Int
end
function allocation(n,id)
 full=CuArray{UInt8,1,CUDA.DeviceMemory}(undef,n*4+512);fill!(full,UInt8(0xa7));CUDA.device_synchronize()
 data=unsafe_wrap(CuArray{Float32,1,CUDA.DeviceMemory},reinterpret(CUDA.CuPtr{Float32},pointer(full)+256),n;own=false)
 Buffer(full,data,n,id)
end
struct Layer{B}
 M::Int;N::Int;O::Int;buffers::Vector{B}
end
function layers()
 result=Any[];id=0
 for(m,n,o) in ((0,0,784),(25,6,3456),(16,1,216),(216,10,10))
  bs=[allocation(k,id+j-1) for(j,k) in enumerate((o,o,n,m*n,o,o,m*n))];id+=7
  push!(result,Layer(m,n,o,bs))
 end
 result
end
data(l,i)=l.buffers[i].data
function clear!(s,l)
 fill!(data(l,1),0f0);fill!(data(l,2),0f0);s.memsets+=2
end
function bpclear!(s,l)
 for i in (5,6,7);fill!(data(l,i),0f0);end;s.memsets+=3
end
function snapshot!(s,ls,kind,step,label,pred)
 h=Int32[kind,step,label,pred];write(s.science,h);append!(s.output,Float64.(h))
 for(li,l) in enumerate(ls),(fi,b) in enumerate(l.buffers)
  if b.n>0 && !(li==1 && fi in(5,6))
   a=Array(b.data);write(s.science,a);append!(s.output,Float64.(a))
  end
 end
end
function setoutput!(s,ls,input)
 if s.capture&&s.pending
  paused(s) do;snapshot!(s,ls,1,s.oldstep,s.oldlabel,-1);s.pending=false;end
 end
 copyto!(data(ls[1],1),input);CUDA.device_synchronize()
 if s.capture;paused(s) do;append!(s.prefix,Float64.(Array(data(ls[1],1))));end;end
end
function forward!(s,ls,image)
 for l in ls;clear!(s,l);end
 input=Float32.(image);setoutput!(s,ls,input)
 i,c,b,f=ls
 dispatch!(Val(:fp_preact_c1),s,data(i,1),data(c,2),data(c,4));dispatch!(Val(:fp_bias_c1),s,data(c,2),data(c,3));dispatch!(Val(:apply_step_function),s,data(c,2),data(c,1),Int32(c.O))
 dispatch!(Val(:fp_preact_s1),s,data(c,1),data(b,2),data(b,4));dispatch!(Val(:fp_bias_s1),s,data(b,2),data(b,3));dispatch!(Val(:apply_step_function),s,data(b,2),data(b,1),Int32(b.O))
 dispatch!(Val(:fp_preact_f),s,data(b,1),data(f,2),data(f,4));dispatch!(Val(:fp_bias_f),s,data(f,2),data(f,3));dispatch!(Val(:apply_step_function),s,data(f,2),data(f,1),Int32(f.O))
end
function back!(s,ls)
 i,c,b,f=ls
 dispatch!(Val(:bp_weight_f),s,data(f,7),data(f,6),data(b,1));dispatch!(Val(:bp_bias_f),s,data(f,3),data(f,6))
 dispatch!(Val(:bp_output_s1),s,data(b,5),data(f,4),data(f,6));dispatch!(Val(:bp_preact_s1),s,data(b,6),data(b,5),data(b,2));dispatch!(Val(:bp_weight_s1),s,data(b,7),data(b,6),data(c,1));dispatch!(Val(:bp_bias_s1),s,data(b,3),data(b,6))
 dispatch!(Val(:bp_output_c1),s,data(c,5),data(b,4),data(b,6));dispatch!(Val(:bp_preact_c1),s,data(c,6),data(c,5),data(c,2));dispatch!(Val(:bp_weight_c1),s,data(c,7),data(c,6),data(i,1));dispatch!(Val(:bp_bias_c1),s,data(c,3),data(c,6))
 for l in (f,b,c);dispatch!(Val(:apply_grad),s,data(l,4),data(l,7),Int32(l.M*l.N));end
end
function learn!(s,ls,images,labels)
 println("Learning")
 for epoch=1:2
  err=0f0
  for k=1:length(labels)
   s.step=k-1;s.label=labels[k];forward!(s,ls,images[k])
   for l in (ls[4],ls[3],ls[2]);bpclear!(s,l);end
   dispatch!(Val(:makeError),s,data(ls[4],6),data(ls[4],1),Int32(labels[k]),Int32(10))
   r=Array(data(ls[4],6));CUDA.device_synchronize();sum=0f0
   for v in r;sum+=v*v;end
   tmp=sqrt(sum)
   if s.capture
    paused(s) do;snapshot!(s,ls,0,s.step,s.label,-1);s.pending=true;s.oldstep=s.step;s.oldlabel=s.label;end
   end
   err+=tmp;back!(s,ls)
  end
  err/=Float32(length(labels))
  if s.capture;paused(s) do;append!(s.output,(30.,Float64(err)));s.valid&=isfinite(err)&&err>=.01f0;end;end
  @printf("error: %e\n",err)
  if err<.01f0;println("Training complete, error less than threshold");break;end
 end
end
function test!(s,ls,images,labels)
 println("Testing");errors=0
 for k=1:length(labels)
  s.step=k-1;s.label=labels[k];forward!(s,ls,images[k]);res=Array(data(ls[4],1));CUDA.device_synchronize();pred=0
  for j=2:10;if res[pred+1]<res[j];pred=j-1;end;end
  if s.capture;paused(s) do;snapshot!(s,ls,2,s.step,s.label,pred);end;end
  errors+=pred!=labels[k]
 end
 if s.capture;paused(s) do;append!(s.output,(31.,Float64(errors)));end;end
 @printf("Error Rate: %.2f%%\n",100*errors/length(labels))
end
function dataset(folder,prefix)
 images=read(joinpath(folder,prefix*"-images.idx3-ubyte"));labels=read(joinpath(folder,prefix*"-labels.idx1-ubyte"));be(b,o)=foldl((a,x)->(a<<8)|Int(x),b[o:o+3];init=0)
 @assert be(images,1)==2051 && be(labels,1)==2049 && be(images,9)==28 && be(images,13)==28
 n=be(images,5);@assert be(labels,5)==n && length(images)==16+n*784 && length(labels)==8+n && all(<=(9),labels[9:end])
 ([Float64.(images[17+(k-1)*784:16+k*784])./255. for k=1:n],Int.(labels[9:end]))
end
function physical!(s,ls,path)
 open(path,"w") do io
  for l in ls,b in l.buffers
   raw=Array(b.full);write(io,UInt64(b.id),UInt64(b.n*4),raw);s.valid&=all(==(0xa7),raw[1:256])&&all(==(0xa7),raw[end-255:end])
  end
 end
end
function main()
 @assert length(ARGS)==3
 case,initial_path,folder=ARGS;mkpath(folder);initial=open(f->read!(f,Vector{Float32}(undef,2343)),initial_path);@assert filesize(initial_path)==2343*4
 images,labels=dataset(case,"train");testimages,testlabels=dataset(case,"t10k");@assert length(labels)==100&&length(testlabels)==20
 ls=layers();s=State()
 for trial=0:5
  s.capture=trial>0;s.pending=false;s.valid=true;s.seconds=0;s.launches=0;s.memsets=0;empty!(s.prefix);empty!(s.output)
  offset=0
  for l in ls,i in(3,4)
   n=l.buffers[i].n;if n>0;copyto!(data(l,i),initial[offset+1:offset+n]);end;offset+=n
  end
  CUDA.device_synchronize()
  if s.capture
   for l in ls,i in(3,4);append!(s.prefix,Float64.(Array(data(l,i))));end
   append!(s.prefix,Float64.(labels));append!(s.prefix,Float64.(testlabels));s.science=open(joinpath(folder,"$(MODEL)_trial$(trial).science.bin"),"w")
  end
  s.mark=time_ns();s.running=true;learn!(s,ls,images,labels);test!(s,ls,testimages,testlabels);CUDA.device_synchronize();s.seconds+=(time_ns()-s.mark)*1e-9;s.running=false
  if trial==0;continue;end
  close(s.science);physical!(s,ls,joinpath(folder,"$(MODEL)_trial$(trial).physical.bin"));prefix=length(s.prefix);append!(s.prefix,s.output);s.valid&=all(isfinite,s.prefix)&&s.launches==4780&&s.memsets==3560
  path="trial$(trial).bin";write(path,s.prefix)
  println("STUDY_TRIAL "*json(Dict("trial"=>trial,"time_seconds"=>s.seconds,"correct"=>s.valid,"checked_values"=>length(s.prefix),"max_abs_error"=>nothing,"max_rel_error"=>nothing,"input_prefix_values"=>prefix,"authored_kernel_launches"=>s.launches,"memset_calls"=>s.memsets,"timing_region"=>"complete_source_learn2_test_pipeline_observer_split_context_completion","jit_excluded"=>true,"internal_repeats"=>1,"output_artifact"=>Dict("path"=>path,"dtype"=>"float64","count"=>length(s.prefix),"byte_order"=>"little"))))
  flush(stdout);s.valid||error("incomplete/invalid CNN pipeline")
 end
 for l in ls,b in l.buffers;CUDA.unsafe_free!(b.data);CUDA.unsafe_free!(b.full);end
end
