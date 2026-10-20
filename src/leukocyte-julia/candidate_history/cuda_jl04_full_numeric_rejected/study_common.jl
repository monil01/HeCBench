using Printf
json(x::AbstractString)="\""*replace(x,'\\'=>"\\\\",'"'=>"\\\"")*"\""
json(x::Bool)=x ? "true" : "false"
json(x::Nothing)="null"
json(x::Number)=isfinite(x) ? string(x) : "null"
json(x::AbstractVector)="["*join(json.(x),",")*"]"
json(x::AbstractDict)="{"*join((json(string(k))*":"*json(v) for (k,v) in x),",")*"}"
mutable struct StudyState
 capture::Bool;trial::Int;frames::Int;frame::Int;prefix::Vector{Float64};output::Vector{Float64};seconds::Float64;live::Dict{String,Any};raw::IO;guards::IO;layout::IO;timings::IO;valid::Bool
end
function StudyState()
 StudyState(false,0,0,0,Float64[],Float64[],0.0,Dict{String,Any}(),devnull,devnull,devnull,devnull,true)
end
function allocate!(s,name,input::Vector{T}) where T
 @assert !haskey(s.live,name);count=length(input);poison=reinterpret(T,fill(UInt8(165),4))[1]
 host=fill(poison,count+128);host[65:64+count]=input
 full=CuArray(host);data=@view full[65:64+count];s.live[name]=(full=full,data=data,count=count,type=T);data
end
function empty_alloc!(s,name,T,count)
 poison=reinterpret(T,fill(UInt8(165),4))[1];allocate!(s,name,fill(poison,count))
end
function release!(s,name)
 a=s.live[name];physical=collect(reinterpret(UInt8,Array(a.full)));first=physical[1:256];last=physical[end-255:end]
 s.valid&=all(==(165),first)&&all(==(165),last)
 if s.capture
  println(s.layout,json(Dict("kind"=>"release_guard","name"=>name,"frame"=>s.frame,"guard_offset_bytes"=>position(s.guards),"guard_bytes"=>512)))
  write(s.guards,first);write(s.guards,last)
 end
 # Views share allocation; remove then explicit free after all field handles out of scope.
 delete!(s.live,name);CUDA.unsafe_free!(a.full)
end
function capture_gpu!(s,stage,phase)
 logical=Dict{String,Any}();s.capture||return logical;fields=Any[]
 for name in sort!(collect(keys(s.live)))
  a=s.live[name];host=Array(a.full);physical=collect(reinterpret(UInt8,host));payload=copy(host[65:64+a.count]);logical[name]=payload
  first=physical[1:256];last=physical[end-255:end];s.valid&=all(==(165),first)&&all(==(165),last)
  push!(fields,Dict("name"=>name,"dtype"=>a.type==Int32 ? "int32" : "float32","raw_offset_bytes"=>position(s.raw),"physical_bytes"=>length(physical),"payload_offset_bytes"=>256,"payload_bytes"=>a.count*4,"guard_offset_bytes"=>position(s.guards),"guard_bytes"=>512))
  write(s.raw,physical);write(s.guards,first);write(s.guards,last)
 end
 println(s.layout,json(Dict("kind"=>"gpu_buffers","stage"=>stage,"phase"=>phase,"frame"=>s.frame,"buffers"=>fields)));logical
end
function record_frame!(s,frame,cropped,scaled,storage)
 s.frame=frame;s.capture||return;data=@view storage[33:end-32]
 println(s.layout,json(Dict("kind"=>"video_frame","frame"=>frame,"width"=>640,"height"=>480,"cropped"=>Int(cropped),"scaled"=>Int(scaled),"raw_offset_bytes"=>position(s.raw),"payload_bytes"=>length(data),"guard_offset_bytes"=>position(s.guards),"guard_bytes"=>64)))
 write(s.raw,data);write(s.guards,storage[1:32]);write(s.guards,storage[end-31:end]);append!(s.prefix,(frame,640,480,Int(cropped),Int(scaled)))
 for at=1:6:length(data)
  word=UInt64(0);for j=0:min(5,length(data)-at);word|=UInt64(data[at+j])<<(8*j);end;push!(s.prefix,Float64(word))
 end
end
function before_gicov!(s,m,n)
 v=capture_gpu!(s,"GICOV","before");s.capture||return
 append!(s.prefix,(m,n,m-44,n-44,256,cld((m-44)*(n-44),256)))
 for name in ("d_cos_angle","d_grad_x","d_grad_y","d_sin_angle","d_tX","d_tY");append!(s.prefix,v[name]);end
end
function before_dilation!(s,m,n)
 v=capture_gpu!(s,"DILATION","before");s.capture||return
 append!(s.prefix,(25,25,m,n,176,cld(m*n,176)));append!(s.prefix,v["d_strel"])
 append!(s.output,(2,s.frame,m,n));append!(s.output,v["d_gicov"])
end
function before_mgvf!(s,cells)
 v=capture_gpu!(s,"MGVF","before");s.capture||return
 append!(s.prefix,(1.0f0,1.0f0,1.0f-10,1.0f-5,500));append!(s.output,(3,s.frame,cells))
 for name in ("d_I_offsets","d_m_array","d_n_array");append!(s.output,v[name]);end
 s.valid&=v["d_I_all"]==v["d_IMGVF_all"];append!(s.output,v["d_I_all"])
end
function after_kernel!(s,stage,seconds,cells=0)
 s.seconds+=seconds;v=capture_gpu!(s,stage,"after");s.capture||return
 counts=Int32[]
 if stage=="GICOV"
  append!(s.output,(1,s.frame));append!(s.output,v["d_gicov"])
 elseif stage=="DILATION"
  append!(s.output,(2,s.frame));append!(s.output,v["d_img_dilated"])
 else
  append!(s.output,(3,s.frame,cells));append!(s.output,v["d_IMGVF_all"]);counts=v["d_iterations"];s.valid&=all(k->1<=k<=500,counts)
 end
 println(s.timings,json(Dict("stage"=>stage,"frame"=>s.frame,"seconds"=>seconds,"iterations"=>counts)))
end
function record_trajectory!(s,frame,x,y,r)
 s.capture||return;append!(s.output,(5,frame,length(x),20))
 for c=eachindex(x);append!(s.output,(x[c],y[c]));append!(s.output,r[c,:]);end
end
function run_mgvf(s,patches)
 cells=length(patches);ms=Int32[size(p,1) for p in patches];ns=Int32[size(p,2) for p in patches];offsets=Int32[];data=Float32[]
 for p in patches
  push!(offsets,length(data));for r=1:size(p,1),c=1:size(p,2);push!(data,Float32(p[r,c]));end
 end
 dinput=allocate!(s,"d_I_all",data);dout=allocate!(s,"d_IMGVF_all",data);doff=allocate!(s,"d_I_offsets",offsets);dm=allocate!(s,"d_m_array",ms);dn=allocate!(s,"d_n_array",ns);counts=empty_alloc!(s,"d_iterations",Int32,cells)
 CUDA.device_synchronize();before_mgvf!(s,cells);CUDA.device_synchronize();start=time_ns();launch_mgvf!(dout,dinput,doff,dm,dn,counts,cells);CUDA.device_synchronize();elapsed=(time_ns()-start)*1e-9;after_kernel!(s,"MGVF",elapsed,cells)
 actual=Array(dout);all(isfinite,actual)||error("nonfinite flow");all(v->-.01<=v<=1.01,actual)||error("flow sanity range")
 fields=Matrix{Float64}[]
 for c=1:cells
  m=ms[c];n=ns[c];off=offsets[c];field=zeros(m,n)
  for r=0:m-1,j=0:n-1;field[r+1,j+1]=actual[off+r*n+j+1];end;push!(fields,field)
 end
 for name in ("d_IMGVF_all","d_I_all","d_I_offsets","d_m_array","d_n_array","d_iterations");release!(s,name);end
 fields
end
function pipeline!(s,video,frames)
 s.frames=frames;s.frame=0;s.capture&&push!(s.prefix,frames)
 image=video_frame(video,0,true,false,s);gx,gy=gradients(image);m,n=size(image)
 theta=[Float32(Float64(i)*2.0*SOURCE_PI/150.0) for i=0:149];si=csinf.(theta);co=ccosf.(theta)
 tx=Int32[];ty=Int32[]
 for k=0:6,p=1:150
  radius=Float64(8+2*k);push!(tx,trunc(Int32,Float64(ccosf(theta[p]))*radius));push!(ty,trunc(Int32,Float64(csinf(theta[p]))*radius))
 end
 dgx=allocate!(s,"d_grad_x",vec(Float32.(gx)));dgy=allocate!(s,"d_grad_y",vec(Float32.(gy)));dsi=allocate!(s,"d_sin_angle",si);dco=allocate!(s,"d_cos_angle",co);dtx=allocate!(s,"d_tX",tx);dty=allocate!(s,"d_tY",ty);dg=allocate!(s,"d_gicov",zeros(Float32,m*n))
 CUDA.device_synchronize();before_gicov!(s,m,n);CUDA.device_synchronize();start=time_ns();launch_gicov!(dgx,dgy,dsi,dco,dtx,dty,dg,m,n);CUDA.device_synchronize();elapsed=(time_ns()-start)*1e-9;after_kernel!(s,"GICOV",elapsed)
 gicov=reshape(Array(dg),m,n)
 st=Float32[];for i=0:24,j=0:24;push!(st,sqrt(Float32((i-12)^2+(j-12)^2))<=12 ? 1.0f0 : 0.0f0);end
 dst=allocate!(s,"d_strel",st);dd=empty_alloc!(s,"d_img_dilated",Float32,m*n)
 CUDA.device_synchronize();before_dilation!(s,m,n);CUDA.device_synchronize();start=time_ns();launch_dilation!(dst,dg,dd,m,n);CUDA.device_synchronize();elapsed=(time_ns()-start)*1e-9;after_kernel!(s,"DILATION",elapsed)
 dilation=permutedims(reshape(Array(dd),n,m));centers=detect_centers(gicov,dilation,gx,gy)
 if s.capture
  append!(s.output,(4,length(centers)));for (x,y) in centers;append!(s.output,(x,y));end
 end
 if length(centers)!=36
  folder=ARGS[2];base=joinpath(folder,"$(MODEL)_detector_rejection_$(getpid())")
  for (name,values) in (("gx",vec(Float32.(gx))),("gy",vec(Float32.(gy))),("si",si),("co",co),("tx",tx),("ty",ty),("gicov",vec(gicov)),("dilation",vec(permutedims(dilation))))
   write(base*"_"*name*".bin",values)
  end
  write(base*".json",json(Dict("cells"=>length(centers),"centers"=>[collect(c) for c in centers],"input_source"=>"actual own Julia warmup inputs and own GPU readbacks; diagnostic only")))
  error("detected $(length(centers)) cells, expected36; actual diagnostic fields retained at $base")
 end
 for name in ("d_sin_angle","d_cos_angle","d_tX","d_tY","d_grad_x","d_grad_y","d_gicov","d_strel","d_img_dilated");release!(s,name);end
 track_pipeline!(s,video,centers,frames,run_mgvf)
end
function main()
 length(ARGS)>=2||error("video and persistent evidence folder required");folder=ARGS[2];mkpath(folder)
 pilot=length(ARGS)==3&&ARGS[3]=="pilot";s=StudyState();video=ccall((:leuko_video_open,VIDEO_LIB),Ptr{Cvoid},(Cstring,),ARGS[1]);video!=C_NULL||error("AVI open")
 try
  for trial=0:(pilot ? 1 : 5)
   s.capture=trial>0;s.trial=trial;empty!(s.prefix);empty!(s.output);s.seconds=0.0;s.valid=true;started=time_ns()
   for frames in (1,3,100)
    if s.capture
     base=joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_case$(frames)")
     s.raw=open(base*".raw.bin","w");s.guards=open(base*".guards.u8","w");s.layout=open(base*".layout.jsonl","w");s.timings=open(base*".timings.jsonl","w")
    end
    pipeline!(s,video,frames);isempty(s.live)||error("allocation lifetime")
    if s.capture;close(s.raw);close(s.guards);close(s.layout);close(s.timings);end
   end
   trial==0&&continue
   prefix=length(s.prefix);@assert prefix==6329529;append!(s.prefix,s.output);s.valid&=all(isfinite,s.prefix)
   artifact="trial$(trial).bin";write(artifact,s.prefix)
   row=Dict("trial"=>trial,"time_seconds"=>s.seconds,"correct"=>s.valid,"checked_values"=>length(s.prefix),"max_abs_error"=>nothing,"max_rel_error"=>nothing,"input_prefix_values"=>prefix,"instrumented_driver_wall_seconds"=>(time_ns()-started)*1e-9,"timing_region"=>"sum_original_3stage_kernel_ROIs_1_3_100frames_including_iteration_counter_store","jit_excluded"=>true,"internal_repeats"=>1,"output_artifact"=>Dict("path"=>artifact,"dtype"=>"float64","count"=>length(s.prefix),"byte_order"=>"little"))
   println("STUDY_TRIAL "*json(row));flush(stdout);s.valid||error("guard/finite/control check failed")
  end
 finally
  ccall((:leuko_video_close,VIDEO_LIB),Cint,(Ptr{Cvoid},),video)
 end
end
