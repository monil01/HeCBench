const ROOT=@__DIR__
const CHECKLIB=joinpath(ROOT,"study_check.so")
mutable struct HState{B,D,P}
 ci::Int;n::Int;m::Int;buffers::B;device::D;strides::P
end
function alloc(::Type{T},n)where T
 storage=CuArray{T}(undef,n+64);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(storage)),UInt8(165),sizeof(T)*length(storage));(storage=storage,data=view(storage,33:n+32),n=n,total=n+64,pitched=false,stride=0,rows=1)
end
function alloc_pitch(::Type{T},n)where T
 stride=cld(n*sizeof(T),512)*512÷sizeof(T);total=stride*82;storage=CuArray{T}(undef,total);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(storage)),UInt8(165),sizeof(T)*total);(storage=storage,data=view(storage,stride+1:stride*81),n=n,total=total,pitched=true,stride=stride,rows=80)
end
function states()
 nc=ccall((:hbc_ncases,CHECKLIB),Cint,());ss=[]
 for ci in 0:nc-1
  n=Int(ccall((:hbc_field,CHECKLIB),Cint,(Cint,Cint),ci,0));m=Int(ccall((:hbc_field,CHECKLIB),Cint,(Cint,Cint),ci,1))
  bs=(R=alloc(Int32,n+1),C=alloc(Int32,2*m),F=alloc(Int32,2*m),next=alloc(Int32,1),jia=alloc(Int32,1),diam=alloc(Int32,512),source=alloc(Int32,0),bc=alloc(Float32,n),dist=alloc_pitch(Int32,n),Q=alloc_pitch(Int32,n),Q2=alloc_pitch(Int32,n),S=alloc_pitch(Int32,n),ep=alloc_pitch(Int32,n+1),sigma=alloc_pitch(UInt64,n),delta=alloc_pitch(Float32,n))
  for(field,key,len)in ((0,:R,n+1),(1,:C,2*m),(2,:F,2*m))
   p=ccall((:hbc_array,CHECKLIB),Ptr{Int32},(Cint,Cint),ci,field);copyto!(getproperty(bs,key).data,copy(unsafe_wrap(Vector{Int32},p,len)))
  end
  device=map(b->b.data,bs);pitch=(dist=Int32(bs.dist.stride),sigma=Int32(bs.sigma.stride),delta=Int32(bs.delta.stride),Q=Int32(bs.Q.stride),Q2=Int32(bs.Q2.stride),S=Int32(bs.S.stride),ep=Int32(bs.ep.stride));push!(ss,HState(ci,n,m,bs,device,pitch))
 end
 ss
end
function reset!(s)
 for name in (:bc,:jia,:diam)
  b=getproperty(s.buffers,name);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(b.data)),UInt8(0),b.n*sizeof(eltype(b.storage)))
 end
 copyto!(s.buffers.next.data,Int32[80])
 for name in (:dist,:Q,:Q2,:S,:ep,:sigma,:delta)
  b=getproperty(s.buffers,name);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(b.storage)),UInt8(165),b.total*sizeof(eltype(b.storage)))
 end
 CUDA.device_synchronize();nothing
end
function prefix(s)
 out=Float64[]
 for k in (:R,:C,:F,:bc,:next,:jia,:diam);append!(out,Array(getproperty(s.device,k)));end
 out
end
function run(s)
 CUDA.device_synchronize();start=time_ns();launch_hbc(s);CUDA.device_synchronize();(time_ns()-start)*1e-9
end
function capture(s,path)
 raw_arrays=map(b->Array(b.storage),s.buffers);logical=map((a,b)->b.pitched ? view(a,b.stride+1:b.stride*81) : view(a,33:b.n+32),raw_arrays,s.buffers)
 ma=Ref{Cdouble}(0);mr=Ref{Cdouble}(0);bc=copy(logical.bc);ok=ccall((:hbc_check_bc,CHECKLIB),Cint,(Cint,Ptr{Float32},Ptr{Cdouble},Ptr{Cdouble}),s.ci,bc,ma,mr)==1
 # Fullactualhostrows copied from retainedstorage, with physicalpitch andunreadtail preserved.
 dist=copy(logical.dist);sigma=copy(logical.sigma);delta=copy(logical.delta);Q=copy(logical.Q);Q2=copy(logical.Q2);S=copy(logical.S);ep=copy(logical.ep);diam=copy(logical.diam);next=only(logical.next);jia=only(logical.jia)
 p=s.strides
 rowok=ccall((:hbc_check_rows,CHECKLIB),Cint,(Cint,Cint,Ptr{Int32},Csize_t,Ptr{UInt64},Csize_t,Ptr{Float32},Csize_t,Ptr{Int32},Csize_t,Ptr{Int32},Csize_t,Ptr{Int32},Csize_t,Ptr{Int32},Csize_t,Ptr{Int32},Cint,Cint),s.ci,80,dist,p.dist,sigma,p.sigma,delta,p.delta,Q,p.Q,Q2,p.Q2,S,p.S,ep,p.ep,diam,next,jia)==1
 ok=ok&&rowok;bytes=UInt8[];guards=UInt8[];layouts=String[]
 for name in keys(s.buffers)
  b=getproperty(s.buffers,name);a=getproperty(raw_arrays,name);piece=reinterpret(UInt8,a);offset=length(bytes);guardoff=length(guards);append!(bytes,piece);segments=Tuple{Int,Int}[];width=sizeof(eltype(a))
  if b.pitched
   push!(segments,(0,b.stride*width))
   for row in 1:80;push!(segments,((row*b.stride+b.n)*width,(b.stride-b.n)*width));end
   push!(segments,(81*b.stride*width,b.stride*width))
  else
   push!(segments,(0,32*width));push!(segments,((32+b.n)*width,32*width))
  end
  for(off,count)in segments;append!(guards,view(piece,off+1:off+count));end
  typename=eltype(a)==Float32 ? "float32" : eltype(a)==UInt64 ? "uint64" : "int32"
  native_name=name==:dist ? "d" : string(name)
  segs=join(("[$(x),$(y)]" for(x,y)in segments),",")
  push!(layouts,"{\"name\":\"$(native_name)\",\"dtype\":\"$(typename)\",\"offset_bytes\":$(offset),\"count_elements\":$(length(a)),\"element_bytes\":$(width),\"logical_width_elements\":$(b.n),\"rows\":$(b.rows),\"stride_elements\":$(b.stride),\"pitched\":$(b.pitched),\"guard_offset_bytes\":$(guardoff),\"guard_count_bytes\":$(length(guards)-guardoff),\"guard_segments\":[$(segs)]}")
 end
 open(io->write(io,bytes),path*".raw.bin","w");open(io->write(io,guards),path*".guards.u8","w");open(io->write(io,"{\"buffers\":["*join(layouts,",")*"]}"),path*".layout.json","w")
 guardok=all(==(UInt8(165)),guards);out=Float64.(bc.*0.5f0);push!(out,next);push!(out,guardok ? 0 : 1)
 println("STUDY_RAW_ARTIFACT {\"model\":\"",MODEL,"\",\"trial_case\":\"",path,"\",\"raw_bytes\":",length(bytes),",\"guard_bytes\":",length(guards),"}")
 println(stderr,"case",s.ci," n",s.n," next",next," jia",jia," independent",ok," guards",guardok)
 (output=out,correct=ok&&guardok,maxabs=ma[],maxrel=mr[])
end
function main()
 length(ARGS)==1||error("persistent raw evidence folder required");folder=ARGS[1];mkpath(folder);ss=states()
 for s in ss;reset!(s);run(s);end
 for trial in 1:5
  pre=Float64[];output=Float64[];seconds=0.0;ma=0.0;mr=0.0;ok=true
  for s in ss
   reset!(s);append!(pre,prefix(s));seconds+=run(s);result=capture(s,joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_case$(s.ci)"));append!(output,result.output);ok&=result.correct;ma=max(ma,result.maxabs);mr=max(mr,result.maxrel)
  end
  append!(pre,output);path="trial$(trial).bin";open(io->write(io,pre),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",length(pre),",\"max_abs_error\":",ma,",\"max_rel_error\":",mr,",\"timing_region\":\"sum4_full_exact_Brandes_source_kernel_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(pre),",\"byte_order\":\"little\"}}")
  ok||error("independent fullBC/workspace/guard check rejected")
 end
end
