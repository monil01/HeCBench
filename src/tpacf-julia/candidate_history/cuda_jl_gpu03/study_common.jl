using Libdl,Printf,SHA
include(joinpath(@__DIR__,"study_cases.jl"))
const ROOT=normpath(@__DIR__)
const OBSLIB=Libdl.dlopen(joinpath(pwd(),"study_observer.so"))
const TP_START=Libdl.dlsym(OBSLIB,:tp_start);const TP_BEGIN=Libdl.dlsym(OBSLIB,:tp_begin)
const TP_CASE=Libdl.dlsym(OBSLIB,:tp_case);const TP_INPUT=Libdl.dlsym(OBSLIB,:tp_input)
const TP_GUARD=Libdl.dlsym(OBSLIB,:tp_guard);const TP_OUTPUT=Libdl.dlsym(OBSLIB,:tp_output)
const TP_WRITE=Libdl.dlsym(OBSLIB,:tp_write);const TP_FINISH=Libdl.dlsym(OBSLIB,:tp_finish)
const G=16384
function guarded(::Type{T},n) where T
 raw=CuArray{UInt8}(undef,n*sizeof(T)+512);fill!(raw,UInt8(0xa5));d=unsafe_wrap(CuArray,convert(CuPtr{T},pointer(raw)+256),n;own=false)
 CUDA.memset(convert(CuPtr{UInt8},pointer(d)),UInt8(0),n*sizeof(T));(;raw,d,n)
end
function read_catalog(ci,k,n)
 padded=cld(n,G)*G;x=zeros(Float64,padded);y=similar(x);fill!(y,0.0);z=copy(y);count=0
 open(joinpath(ROOT,"data","case$(ci)_catalog$(k).txt")) do io
  for line in eachline(io)
   startswith(line,"#")&&continue;count+=1;count>n&&break;v=split(line);length(v)==5||error("named fivecolumncatalog")
   x[count]=parse(Float64,v[3]);y[count]=parse(Float64,v[4]);z[count]=parse(Float64,v[5])
  end
 end
 count==n||error("catalog completeinput");(x,y,z)
end
function source_jk_layout(c)
 nk=cld(c.n,G);d=zeros(Int32,c.njk,nk);r=similar(d);fill!(r,Int32(0));current=1;written=0
 for i in 1:nk
  remainder=G
  while remainder>0&&current<=c.njk
   if remainder<c.jk[current]-written
    d[current,i]+=Int32(remainder);written+=remainder;remainder=0
   else
    remainder-=c.jk[current]-written;d[current,i]+=c.jk[current]-Int32(written);current+=1;written=0
   end
  end
 end
 for i in 1:nk-1;r[1,i]=Int32(G);end
 r[1,nk]=Int32(c.n%G==0 ? G : c.n%G);d,r
end
function upload_tile!(c,which,host,tile)
 buffers=which==1 ? c.input1 : c.input2
 for axis in 1:3
  first=tile*G+1;copyto!(buffers[axis].d,1,host[axis],first,G);CUDA.device_synchronize()
  start=time_ns();actual=Array(buffers[axis].d);ccall(TP_INPUT,Cvoid,(Ptr{Float64},Csize_t),actual,sizeof(actual));c.paused[]+=(time_ns()-start)*1e-9
 end
 nothing
end
function GPUHistogram!(c,sub,base,startwords,assignments)
 words=assignments÷4;blocks=cld(words,8064);blocks<=16384||error("sourcehistogramlimit")
 hist_launch!(c,startwords,words,blocks);blocks>32&&merge_launch!(c,blocks);nend=min(32,blocks);host=c.partial;copyto!(host,1,c.histo.d,1,nend*64);CUDA.device_synchronize()
 @inbounds for i in 2:nend,j in 1:64;host[j]+=host[(i-1)*64+j];end
 @inbounds for j in 1:64;sub[base+j]=host[j];end
 nothing
end
function source_symm!(c,type,host,layout,histo,launch)
 sub=Vector{UInt32}(undef,c.case.njk*64);nk=cld(c.case.n,G)
 for i in 0:nk-1
  upload_tile!(c,1,host,i);launch();index=0;fill!(sub,UInt32(0))
  for k in 1:c.case.njk
   rows=Int(layout[k,i+1])
   if rows!=0;GPUHistogram!(c,sub,(k-1)*64,index,rows*G);index+=rows*(G÷4);end
  end
  @inbounds for k in 1:(type==1 ? 64 : c.case.njk*64);histo[k]+=Int64(sub[k]);end
  for j in i+1:nk-1
   upload_tile!(c,2,host,j);c.cross();index=0;fill!(sub,UInt32(0))
   for k in 1:c.case.njk
    rows=Int(layout[k,i+1])
    if rows!=0;GPUHistogram!(c,sub,(k-1)*64,index,rows*G);index+=rows*(G÷4);end
   end
   @inbounds for k in 1:(type==1 ? 64 : c.case.njk*64);histo[k]+=Int64(sub[k]);end
  end
 end
 nothing
end
function source_cross!(c,data,random,layout,histo)
 sub=Vector{UInt32}(undef,c.case.njk*64);nk=cld(c.case.n,G)
 for i in 0:nk-1
  upload_tile!(c,1,data,i)
  for j in 0:nk-1
   upload_tile!(c,2,random,j);c.cross();index=0;fill!(sub,UInt32(0))
   for k in 1:c.case.njk
    rows=Int(layout[k,i+1])
    if rows!=0;GPUHistogram!(c,sub,(k-1)*64,index,rows*G);index+=rows*(G÷4);end
   end
   @inbounds for k in 1:c.case.njk*64;histo[k]+=Int64(sub[k]);end
  end
 end
 nothing
end
function timed_call(f,c,args...)
 start=time_ns();before=c.paused[];f(c,args...);(time_ns()-start)*1e-9-(c.paused[]-before)
end
function source_compile(dds,drs,rrs,j,nr)
 dd=zeros(Int64,64,j+1);dr=similar(dd);fill!(dr,0);rr=zeros(Int64,64)
 @inbounds for bin in 1:64
  for k in 1:j;dd[bin,1]+=dds[(k-1)*64+bin];end
  for r in 1:nr
   for k in 1:j;dr[bin,1]+=drs[(r-1)*j*64+(k-1)*64+bin];end
   rr[bin]+=rrs[(r-1)*64+bin]
  end
  for k in 1:j
   dd[bin,k+1]=dd[bin,1]-dds[(k-1)*64+bin];dr[bin,k+1]=dr[bin,1]
   for r in 1:nr;dr[bin,k+1]-=drs[(r-1)*j*64+(k-1)*64+bin];end
  end
 end
 dd,dr,rr
end
function source_write_report(path,dd,dr,rr,j,nr)
 open(path,"w") do io
  print(io,"boundary DD DRS RRS w ");for l in 1:j;@printf(io,"DD[%i] DRS[%i] RRS w[%i]  ",l,l,l);end;println(io)
  @printf(io,"<%f %d %d %d -\n",1.0,dd[1,1],dr[1,1],rr[1])
  for k in 1:20
   bb=10.0^(k/5.0);@printf(io,"%f ",bb);bin=k+11
   for l in 1:j+1
    if rr[bin]!=0
     w=(nr*Float64(dd[bin,l])-Float64(dr[bin,l]))/Float64(rr[bin])+1.0
     @printf(io,"%d %d %d %f ",dd[bin,l],dr[bin,l],rr[bin],w)
    else
     @printf(io,"%d %d %d - ",dd[bin,l],dr[bin,l],rr[bin])
    end
   end
   println(io)
  end
  @printf(io,">%f %d %d %d\n",10000.0,dd[22,1],dr[22,1],rr[22])
 end
 nothing
end
function study()
 CUDA.allowscalar(false);captures=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/tpacf_primary"));ccall(TP_START,Cvoid,(Cstring,Cstring,Cstring),MODEL,ROOT,captures)
 constants=collect(reinterpret(Float64,read(joinpath(ROOT,"data/source_boundaries63.bin"))));aux=joinpath(captures,MODEL,"module_constants");mkpath(aux);nonce=string(time_ns())
 for trial in 0:5
  ccall(TP_BEGIN,Cvoid,(Cint,),trial);seconds=0.0
  for (index,case) in enumerate(CASES)
   ci=index-1;input1=ntuple(_->guarded(Float64,G),3);input2=ntuple(_->guarded(Float64,G),3);packed=guarded(UInt32,G*G÷4);histo=guarded(UInt32,16384*64);buffers=(input1...,input2...,packed,histo)
   c=(;case,input1,input2,packed,histo,paused=Ref(0.0),partial=Vector{UInt32}(undef,2048))
   symbols,sym,cross=prepare_dispatch(c);c=merge(c,(;sym,cross));actual_constants=[]
   for symbol in symbols;upload_bins(symbol,constants);CUDA.device_synchronize();actual=read_bins(symbol);reinterpret(UInt64,actual)==reinterpret(UInt64,constants)||error("actualconstantcopybits");push!(actual_constants,actual);end
   # Julia has one module per source ACF entry; retain both complete actual copies.
   file=joinpath(aux,"$(nonce)_trial$(trial)_case$(ci).bin");open(io->foreach(a->write(io,a),actual_constants),file,"w");println("TPACF_CONSTANT_COPIES trial=",trial," case=",ci," modules=",length(symbols)," path=",file," sha256=",bytes2hex(sha256(read(file))))
   ccall(TP_CASE,Cvoid,(Cint,Cint,Cint,Cint,Ptr{Int32},Cint,Ptr{Float64}),ci,case.n,case.nr,case.njk,case.jk,29,actual_constants[1])
   dlayout,rlayout=source_jk_layout(case);dds=zeros(Int64,case.njk*64);drs=zeros(Int64,case.njk*case.nr*64);rrs=zeros(Int64,case.nr*64)
   data=read_catalog(ci,0,case.n);seconds+=timed_call(source_symm!,c,0,data,dlayout,dds,c.sym)
   for r in 1:case.nr
    random=read_catalog(ci,r,case.n);seconds+=timed_call(source_cross!,c,data,random,dlayout,view(drs,(r-1)*case.njk*64+1:r*case.njk*64));seconds+=timed_call(source_symm!,c,1,random,rlayout,view(rrs,(r-1)*64+1:r*64),c.sym)
   end
   pad=cld(case.n,G)*G-case.n
   @inbounds for r in 1:case.nr
    rrs[(r-1)*64+30]-=pad*case.n
    for k in 1:case.njk;drs[(r-1)*case.njk*64+(k-1)*64+30]-=pad*case.jk[k];end
   end
   @inbounds for k in 1:case.njk;dds[(k-1)*64+30]-=pad*case.jk[k];end
   CUDA.device_synchronize()
   for (id,b) in enumerate(buffers)
    ghead=Array(view(b.raw,1:256));gtail=Array(view(b.raw,length(b.raw)-255:length(b.raw)));g=vcat(ghead,gtail);ccall(TP_GUARD,Cvoid,(Cint,Ptr{UInt8}),id-1,g)
   end
   dd,dr,rr=source_compile(dds,drs,rrs,case.njk,case.nr);report="case$(ci)_trial$(trial).txt";source_write_report(report,dd,dr,rr,case.njk,case.nr)
   raw=vcat(dds,drs,rrs);compiled=vcat(vec(dd),vec(dr),rr);ccall(TP_OUTPUT,Cvoid,(Ptr{Int64},Ptr{Int64},Cstring),raw,compiled,report)
   for b in buffers;CUDA.unsafe_free!(b.d);CUDA.unsafe_free!(b.raw);end
  end
  trial>0&&ccall(TP_WRITE,Cvoid,(Cint,Cdouble),trial,seconds)
 end
 ccall(TP_FINISH,Cvoid,());nothing
end
