
using Printf,SHA,Libdl
const G=16;const FG=12345.5f0;const POISON=-23456.5f0;const IG=UInt64(0xa5a55a5a);const R=10
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBSHA="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298"
const CASES=[(n=1,mode=0,nnz=1,ao=0,xo=1,vo=2,rpo=0,co=2,ro=3,oo=0,),(n=17,mode=1,nnz=279,ao=3,xo=292,vo=309,rpo=4,co=22,ro=301,oo=1,),(n=33,mode=2,nnz=62,ao=588,xo=1677,vo=1710,rpo=580,co=614,ro=676,oo=18,),(n=257,mode=3,nnz=744,ao=1772,xo=67821,vo=68078,rpo=738,co=996,ro=1740,oo=51,),(n=1024,mode=4,nnz=30135,ao=68822,xo=1117398,vo=1118422,rpo=2484,co=3509,ro=33644,oo=308,)]
readtyped(T,n)=collect(reinterpret(T,read(joinpath(@__DIR__,n))))
guarded(v,o,n,g)=vcat(fill(g,G),v[o+1:o+n],fill(g,G))
words(v::Vector{Float32})=UInt64.(reinterpret(UInt32,v))
words(v::Vector{UInt64})=v
function capture!(out,s)
 for name in (:da,:dx,:dv,:drp,:dco,:dro);append!(out,words(Array(getproperty(s,name))));end
 for y in s.dy;append!(out,words(Array(y)));end
end
function run_study(launch)
 bytes2hex(sha256(read(LIBPATH)))==LIBSHA||error("libraryhash");lib=Libdl.dlopen(LIBPATH);sym(n)=Libdl.dlsym(lib,n);ck(x)=x==0||error("cuSPARSE status $x")
 create=sym(:cusparseCreate);destroy=sym(:cusparseDestroy);csr=sym(:cusparseCreateCsr);coo=sym(:cusparseCreateCoo);dn=sym(:cusparseCreateDnVec);dmat=sym(:cusparseDestroySpMat);dvec=sym(:cusparseDestroyDnVec);fn=sym(:cusparseSpMV);bufsize=sym(:cusparseSpMV_bufferSize)
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),fn,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==realpath(LIBPATH)||error("functionresolution")
 h=Ref{Ptr{Cvoid}}();ck(ccall(create,Cint,(Ref{Ptr{Cvoid}},),h));version=Ref{Cint}();pm=Ref{UInt32}();stream=Ref{Ptr{Cvoid}}();ck(ccall(sym(:cusparseGetVersion),Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));ck(ccall(sym(:cusparseGetPointerMode),Cint,(Ptr{Cvoid},Ref{UInt32}),h[],pm));ck(ccall(sym(:cusparseGetStream),Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));pm[]==0&&stream[]==C_NULL||error("policy");println("STUDY_VENDOR SpMV=",resolved," sha256=",LIBSHA," version=",version[]," stream=0 pointermode=0")
 f=readtyped(Float32,"input_f32.bin");ix=readtyped(UInt64,"indices_u64.bin");oracle=readtyped(Float32,"oracle_f32.bin");alpha=Ref{Cfloat}(1);beta=Ref{Cfloat}(0);states=Any[];prefix=UInt64[]
 function vendor(s,k)
  GC.@preserve s alpha beta begin
   ck(ccall(fn,Cint,(Ptr{Cvoid},Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,CUDA.CuPtr{Cvoid}),h[],0,alpha,s.mat[k],s.vx,beta,s.vy[k],0,0,pointer(s.buf[k])))
  end
 end
 for c in CASES
  n=c.n;nnz=c.nnz;a=guarded(f,c.ao,n*n,FG);x=guarded(f,c.xo,n,FG);v=guarded(f,c.vo,nnz,FG);rp=guarded(ix,c.rpo,n+1,IG);co=guarded(ix,c.co,nnz,IG);ro=guarded(ix,c.ro,nnz,IG);y=vcat(fill(FG,G),fill(POISON,n),fill(FG,G));da=CuArray(a);dx=CuArray(x);dv=CuArray(v);drp=CuArray(rp);dco=CuArray(co);dro=CuArray(ro);dy=[CuArray(y)for _ in 1:3];mat=Ptr{Cvoid}[];vy=Ptr{Cvoid}[];buf=CuArray{UInt8,1}[];vx=Ref{Ptr{Cvoid}}()
  ck(ccall(dn,Cint,(Ref{Ptr{Cvoid}},Int64,CUDA.CuPtr{Cvoid},Cint),vx,n,pointer(dx,G+1),0))
  GC.@preserve da dx dv drp dco dro dy alpha beta begin
   for k in 1:2
    m=Ref{Ptr{Cvoid}}();yd=Ref{Ptr{Cvoid}}()
    if k==1
     ck(ccall(csr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),m,n,n,nnz,pointer(drp,G+1),pointer(dco,G+1),pointer(dv,G+1),3,3,0,0))
    else
     ck(ccall(coo,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint),m,n,n,nnz,pointer(dro,G+1),pointer(dco,G+1),pointer(dv,G+1),3,0,0))
    end
    ck(ccall(dn,Cint,(Ref{Ptr{Cvoid}},Int64,CUDA.CuPtr{Cvoid},Cint),yd,n,pointer(dy[k+1],G+1),0));bytes=Ref{Csize_t}();ck(ccall(bufsize,Cint,(Ptr{Cvoid},Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,Ref{Csize_t}),h[],0,alpha,m[],vx[],beta,yd[],0,0,bytes));push!(mat,m[]);push!(vy,yd[]);push!(buf,CuArray{UInt8}(undef,max(Int(bytes[]),1)))
   end
  end
  s=(c=c,n=n,a=a,x=x,v=v,rp=rp,co=co,ro=ro,y=y,da=da,dx=dx,dv=dv,drp=drp,dco=dco,dro=dro,dy=dy,mat=mat,vx=vx[],vy=vy,buf=buf);capture!(prefix,s);CUDA.device_synchronize();launch(s.da,s.dx,s.dy[1],n);vendor(s,1);vendor(s,2);CUDA.device_synchronize();push!(states,s)
 end
 for trial in 1:5
  art=copy(prefix);ok=true;seconds=0.;components=Float64[]
  for s in states
   for y in s.dy;copyto!(y,s.y);end;CUDA.device_synchronize()
   for k in 1:3
    start=time_ns()
    for _ in 1:R
     if k==1;launch(s.da,s.dx,s.dy[1],s.n);else;vendor(s,k-1);end
    end
    CUDA.device_synchronize();t=(time_ns()-start)*1e-9/R;seconds+=t;push!(components,t)
   end
   for (dev,host) in [(s.da,s.a),(s.dx,s.x),(s.dv,s.v),(s.drp,s.rp),(s.dco,s.co),(s.dro,s.ro)];ok&=Array(dev)==host;end
   expected=vcat(fill(FG,G),oracle[s.c.oo+1:s.c.oo+s.n],fill(FG,G));for y in s.dy;ok&=Array(y)==expected;end;capture!(art,s)
  end
  path="spmv_trial$trial.bin";open(path,"w")do io;write(io,art);end
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_five_cases_three_dense_CSR_COO_batches_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),seconds,path,length(art),length(art))
 end
 for s in states
  for k in 1:2;ck(ccall(dmat,Cint,(Ptr{Cvoid},),s.mat[k]));ck(ccall(dvec,Cint,(Ptr{Cvoid},),s.vy[k]));end;ck(ccall(dvec,Cint,(Ptr{Cvoid},),s.vx))
 end
 ck(ccall(destroy,Cint,(Ptr{Cvoid},),h[]))
end
