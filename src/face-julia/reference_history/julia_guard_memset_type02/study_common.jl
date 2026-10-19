const CHECKLIB=joinpath(@__DIR__,"study_check.so")
const FEATURE_COUNTER=Ref(0f0)
const ITER_COUNTER=Ref(0)
myround(x)=trunc(Int,x+(x>=0 ? .5 : -.5))
function fixtures()
 n=ccall((:face_ncases,CHECKLIB),Cint,());cs=[]
 for k in 0:n-1
  w=Int(ccall((:face_dim,CHECKLIB),Cint,(Cint,Cint),k,0));h=Int(ccall((:face_dim,CHECKLIB),Cint,(Cint,Cint),k,1));ni=ccall((:face_ninput,CHECKLIB),Cint,(Cint,),k);ptr=ccall((:face_input,CHECKLIB),Ptr{Int32},(Cint,),k);input=copy(unsafe_wrap(Vector{Int32},ptr,ni));pos=w*h
  take(n)=(v=input[pos+1:pos+n];pos+=n;v)
  stages=take(25);rects=take(2913*12);weights=take(2913*3);trees=take(2913);a1=take(2913);a2=take(2913);sth=take(25)
  push!(cs,(w=w,h=h,img=UInt8.(input[1:w*h]),stages=stages,rects=rects,weights=weights,trees=trees,a1=a1,a2=a2,sth=sth,input=input))
 end
 cs
end
function galloc(::Type{T},n)where T
 g=32÷sizeof(T);a=CuArray{T}(undef,n+2g)
 CUDA.memset(pointer(a),UInt8(165),32);CUDA.memset(pointer(a,g+n+1),UInt8(165),32)
 (storage=a,data=view(a,g+1:g+n),g=g,n=n)
end
function gfree!(a,guards)
 # Both original cudaMemcpy guards are synchronous; whole-context completion retains that boundary.
 CUDA.device_synchronize();x=Array(view(a.storage,1:a.g));y=Array(view(a.storage,a.g+a.n+1:length(a.storage)));raw=vcat(collect(reinterpret(UInt8,x)),collect(reinterpret(UInt8,y)))
 all(==(UInt8(165)),raw)||error("GPU allocation guard changed")
 append!(guards,Int64.(raw));CUDA.unsafe_free!(a.storage);nothing
end
function vec!(out,v)
 push!(out,length(v));append!(out,v);nothing
end
function flatten(rs)
 out=Int[];for r in rs;append!(out,r);end;out
end
function group_rectangles(rs)
 n=length(rs);n==0&&return rs;par=collect(1:n)
 function root(i)
  while par[i]!=i;i=par[i];end;i
 end
 for i in 1:n,j in 1:n
  i==j&&continue;a=rs[i];b=rs[j];delta=Float32(Float32(.4)*(min(a[3],b[3])+min(a[4],b[4])))*.5
  if abs(a[1]-b[1])<=delta&&abs(a[2]-b[2])<=delta&&abs(a[1]+a[3]-b[1]-b[3])<=delta&&abs(a[2]+a[4]-b[2]-b[4])<=delta
   par[root(j)]=root(i)
  end
 end
 classes=Vector{Vector{NTuple{4,Int}}}();labels=Dict{Int,Int}()
 for i in 1:n
  rt=root(i);if !haskey(labels,rt);labels[rt]=length(classes)+1;push!(classes,NTuple{4,Int}[]);end;push!(classes[labels[rt]],rs[i])
 end
 av=NTuple{4,Int}[];counts=Int[]
 for v in classes
  inv=1f0/length(v);push!(av,ntuple(k->myround(Float32(sum(a[k] for a in v))*inv),4));push!(counts,length(v))
 end
 out=NTuple{4,Int}[]
 for i in eachindex(av)
  a=av[i];na=counts[i];na<=1&&continue;keep=true
  for j in eachindex(av)
   nb=counts[j];(j==i||nb<=1)&&continue;b=av[j];dx=myround(Float32(b[3])*Float32(.4));dy=myround(Float32(b[4])*Float32(.4))
   if a[1]>=b[1]-dx&&a[2]>=b[2]-dy&&a[1]+a[3]<=b[1]+b[3]+dx&&a[2]+a[4]<=b[2]+b[4]+dy&&(nb>max(3,na)||na<3);keep=false;break;end
  end
  keep&&push!(out,a)
 end
 out
end
@inline function patch_corner(sum,off,po)
 sum[off[1]+po+1]-sum[off[2]+po+1]-sum[off[3]+po+1]+sum[off[4]+po+1]
end
function classify(c,sum,sq,offsets,width,x,y)
 po=y*width+x;corner=(0,23,width*23,width*23+23)
 mean=patch_corner(sum,corner,po)%UInt32;variance=patch_corner(sq,corner,po)%UInt32
 variance=variance*UInt32(576)-mean*mean;norm=variance>0 ? isqrt(variance) : UInt32(1)
 node=0;features=FEATURE_COUNTER[]
 for stage in 0:24
  ss=Int32(0)
  for j in 1:c.stages[stage+1]
   ridx=node*12;wi=node*3
   s=(sum[offsets[ridx+1]+po+1]-sum[offsets[ridx+2]+po+1]-sum[offsets[ridx+3]+po+1]+sum[offsets[ridx+4]+po+1])*c.weights[wi+1]
   s+=(sum[offsets[ridx+5]+po+1]-sum[offsets[ridx+6]+po+1]-sum[offsets[ridx+7]+po+1]+sum[offsets[ridx+8]+po+1])*c.weights[wi+2]
   if offsets[ridx+9]!=-1;s+=(sum[offsets[ridx+9]+po+1]-sum[offsets[ridx+10]+po+1]-sum[offsets[ridx+11]+po+1]+sum[offsets[ridx+12]+po+1])*c.weights[wi+3];end
   # Source signed threshold receives UInt32 normalization converted to int argument.
   threshold=c.trees[node+1]*reinterpret(Int32,norm)
   ss+=s>=threshold ? c.a2[node+1] : c.a1[node+1];node+=1;features+=1f0
  end
  if ss<.4*c.sth[stage+1];FEATURE_COUNTER[]=features;return -stage;end
 end
 FEATURE_COUNTER[]=features;1
end
function detect(c,replay)
 # Source native harness also copies its immutable image fixture inside this declared pipeline.
 image=copy(c.img);small=Vector{UInt8}(undef,c.w*c.h);sum=zeros(Int32,(c.w+1)*(c.h+1));sq=zeros(Int32,length(sum))
 drect=galloc(Int32,length(c.rects));copyto!(drect.data,c.rects);actual_input=Array(drect.data)
 guards=Int64[];trace=Int64[];raw_pointers=UInt64[];candidates=NTuple{4,Int}[];factor=1f0
 while true
  ITER_COUNTER[]+=1;ww=myround(Float32(24)*factor);w=trunc(Int,Float32(c.w)/factor);h=trunc(Int,Float32(c.h)/factor)
  (w<24||h<24)&&break
  xr=(c.w<<16)÷w+1;yr=(c.h<<16)÷h+1
  for y in 0:h-1,x in 0:w-1;small[y*w+x+1]=image[((y*yr)>>16)*c.w+((x*xr)>>16)+1];end
  for y in 0:h-1
   s=Int32(0);q=Int32(0)
   for x in 0:w-1
    v=Int32(small[y*w+x+1]);s+=v;q+=v*v;i=y*w+x+1;sum[i]=s+(y>0 ? sum[i-w] : Int32(0));sq[i]=q+(y>0 ? sq[i-w] : Int32(0))
   end
  end
  if replay;append!(trace,[w,h]);vec!(trace,view(small,1:w*h));vec!(trace,view(sum,1:w*h));vec!(trace,view(sq,1:w*h));end
  dout=galloc(UInt64,2913*12)
  base=UInt64(UInt(pointer(sum)));GC.@preserve sum begin
   launch_address(drect.data,dout.data,base,Int32(w),Int32(2913));CUDA.device_synchronize()
  end
  pointers=Array(dout.data);offsets=Int[]
  pointers[1]==base||error("originalfirstrectangledevicebaseecho differs");append!(raw_pointers,UInt64[replay,w,h,2913,base,pointers[1]])
  for p in pointers
   push!(raw_pointers,p)
   if p==0;push!(offsets,-1);push!(raw_pointers,0);else
    p>=base&&(p-base)%4==0||error("bad normalized host pointer");offset=Int((p-base)÷4);0<=offset<w*h||error("pointer outside integral image")
    base+UInt64(offset)*4==p||error("pointer roundtrip");push!(offsets,offset);push!(raw_pointers,UInt64(unsafe_load(Ptr{Int32}(UInt(p)))))
   end
  end
  replay&&vec!(trace,offsets);gfree!(dout,guards)
  println("detecting faces, iter := ",ITER_COUNTER[]);codes=Int[]
  for x in 0:w-24,y in 0:h-24
   result=classify(c,sum,sq,offsets,w,x,y);replay&&append!(codes,[x,y,result]);result>0&&push!(candidates,(myround(Float32(x)*factor),myround(Float32(y)*factor),ww,ww))
  end
  if replay;vec!(trace,codes);vec!(trace,flatten(candidates));end
  factor*=Float32(1.2)
 end
 result=group_rectangles(candidates);gfree!(drect,guards)
 (rectangles=result,trace=trace,actual_input=actual_input,guards=guards,raw_pointers=raw_pointers)
end
function draw(c,rs)
 im=copy(c.img)
 for(x,y,w,h)in rs
  for i in 0:w-1;im[c.w*y+x+i+1]=255;end
  for i in 0:h-1;im[c.w*(y+i)+x+w+1]=255;end
  for i in 0:w-1;im[c.w*(y+h)+x+w-i+1]=255;end
  for i in 0:h-1;im[c.w*(y+h-i)+x+1]=255;end
 end;im
end
function main()
 length(ARGS)==1||error("require persistent pointer evidence output directory");mkpath(ARGS[1])
 cs=fixtures();for c in cs;detect(c,false);detect(c,true);end;CUDA.device_synchronize()
 for trial in 1:5
  seconds=0.;prefix=Int64[];traces=Int64[];guards=Int64[];raw_pointers=UInt64[]
  for c in cs
   CUDA.device_synchronize();a=time_ns();timed=detect(c,false);CUDA.device_synchronize();seconds+=(time_ns()-a)*1e-9
   append!(prefix,c.input);append!(prefix,timed.actual_input);append!(guards,timed.guards);append!(raw_pointers,timed.raw_pointers)
   replay=detect(c,true);append!(raw_pointers,replay.raw_pointers);replay.rectangles==timed.rectangles||error("timed/replay rectangles differ");vec!(replay.trace,flatten(replay.rectangles));vec!(replay.trace,draw(c,replay.rectangles));append!(traces,replay.trace)
  end
  append!(prefix,traces);append!(prefix,guards);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial$(trial).bin");open(io->write(io,prefix),path,"w");rawpath=joinpath(ARGS[1],"$(MODEL)_$(getpid())_trial$(trial)_u64.bin");open(io->write(io,raw_pointers),rawpath,"w");println("STUDY_POINTER_ARTIFACT {\"trial\":",trial,",\"path\":\"",rawpath,"\",\"dtype\":\"uint64\",\"count\":",length(raw_pointers),"}");ok=ccall((:face_validate,CHECKLIB),Cint,(Ptr{Int64},Int64),prefix,length(prefix))==1
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",length(prefix),",\"max_abs_error\":",ok ? "0" : "null",",\"max_rel_error\":",ok ? "0" : "null",",\"timing_region\":\"allocation_inclusive_detect_pipeline_plus_symmetric_rectangle_input_auditcopy\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int64\",\"count\":",length(prefix),",\"byte_order\":\"little\"}}")
  ok||error("complete face detector independent oracle mismatch")
 end
end
