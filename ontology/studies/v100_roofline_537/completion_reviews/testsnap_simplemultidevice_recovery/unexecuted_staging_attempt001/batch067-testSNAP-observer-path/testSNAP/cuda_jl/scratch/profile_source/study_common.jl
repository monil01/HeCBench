using Libdl
const ROOT="/home/imo/HeCBench_julia/src/testSNAP-julia"
const OBSLIB=Libdl.dlopen(joinpath(pwd(),"study_observer.so"))
const SNAP_START=Libdl.dlsym(OBSLIB,:snap_start)
const SNAP_BEGIN=Libdl.dlsym(OBSLIB,:snap_begin)
const SNAP_CASE=Libdl.dlsym(OBSLIB,:snap_case)
const SNAP_INPUT=Libdl.dlsym(OBSLIB,:snap_input)
const SNAP_GUARD=Libdl.dlsym(OBSLIB,:snap_guard)
const SNAP_OUTPUT=Libdl.dlsym(OBSLIB,:snap_output)
const SNAP_CASE_END=Libdl.dlsym(OBSLIB,:snap_case_end)
const SNAP_WRITE=Libdl.dlsym(OBSLIB,:snap_write)
const SNAP_FINISH=Libdl.dlsym(OBSLIB,:snap_finish)
function fixture(J)
 open(joinpath(ROOT,"data","J$(J)_source_fixture.bin")) do io
  rd(T,n)=read!(io,Vector{T}(undef,n))
  h=rd(UInt32,12);h[1]==0x534e4150||error("fixture magic")
  J,na,ng,nb,ncoeff,nu,nd,nz,ncg,jdim,pdim=Int.(h[2:12]);nt=na+ng;s=rd(Float64,4)
  iu=rd(Int32,jdim);parity=rd(Int32,nu);id=rd(Int32,jdim);iz=rd(Int32,9nz);icg=rd(Int32,jdim^3)
  root=rd(Float64,pdim^2);cg=rd(Float64,ncg);zbeta=rd(Float64,nz);rij=rd(Float64,3na*nb);inside=rd(Int32,na*nb)
  w=rd(Float64,na*nb);rcut=rd(Float64,na*nb);reference=rd(Float64,3nt);coeff=rd(Float64,ncoeff+1);eof(io)||error("fixture tail")
  (;h,s,J,na,ng,nb,ncoeff,nu,nd,nz,ncg,jdim,pdim,nt,iu,parity,id,iz,icg,root,cg,zbeta,rij,inside,w,rcut,reference,coeff)
 end
end
function guarded(::Type{T},n) where T
 raw=CuArray{UInt8}(undef,n*sizeof(T)+512);fill!(raw,UInt8(0xa5));d=unsafe_wrap(CuArray,convert(CuPtr{T},pointer(raw)+256),n;own=false)
 CUDA.memset(convert(CuPtr{UInt8},pointer(d)),UInt8(0),n*sizeof(T));(;raw,d,n)
end
function observe_input(ci,step,id,v)
 ccall(SNAP_INPUT,Cvoid,(Cint,Cint,Cint,Ptr{Cvoid},Csize_t),ci,step,id,v,sizeof(v));nothing
end
function observe_guard(ci,step,id,b)
 # Two actual device reads, not inferred success counters or synthesized bytes.
 g=vcat(Array(view(b.raw,1:256)),Array(view(b.raw,length(b.raw)-255:length(b.raw))))
 ccall(SNAP_GUARD,Cvoid,(Cint,Cint,Cint,Ptr{UInt8}),ci,step,id,g);nothing
end
function study()
 CUDA.allowscalar(false)
 capture=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_capture")
 ccall(SNAP_START,Cvoid,(Cstring,Cstring,Cstring),MODEL,ROOT,capture)
 for trial in 0:5
  ccall(SNAP_BEGIN,Cvoid,(Cint,),trial);seconds=0.0
  for (ci0,(order,steps)) in enumerate(((14,100),(4,3),(2,3)))
   ci=ci0-1;s=fixture(order);na=s.na;nb=s.nb;nt=s.nt
   ccall(SNAP_CASE,Cvoid,(Cint,Ptr{UInt32},Ptr{Float64},Cint),ci,s.h,s.s,steps)
   iu=guarded(Int32,length(s.iu));parity=guarded(Int32,length(s.parity));id=guarded(Int32,length(s.id));iz=guarded(Int32,length(s.iz));icg=guarded(Int32,length(s.icg));root=guarded(Float64,length(s.root));cg=guarded(Float64,length(s.cg));zb=guarded(Float64,length(s.zbeta))
   du=guarded(Float64,na*nb*s.nd*3*2);u=guarded(Float64,na*nb*s.nu*2);d=guarded(Float64,na*nb*3);ut=guarded(Float64,na*s.nu*2);y=guarded(Float64,na*s.nd*2);r=guarded(Float64,na*nb*3);rc=guarded(Float64,na*nb);w=guarded(Float64,na*nb)
   static=(iu,parity,id,iz,icg,root,cg,zb);allbuffers=(static...,du,u,d,ut,y,r,rc,w)
   for (b,v) in zip(static,(s.iu,s.parity,s.id,s.iz,s.icg,s.root,s.cg,s.zbeta));copyto!(b.d,v);end
   CUDA.device_synchronize()
   for (bid,b) in enumerate(static);observe_input(ci,0,bid-1,Array(b.d));end
   observe_input(ci,0,12,s.reference);observe_input(ci,0,13,s.coeff)
   # Reconstruct original AoS host source once. Every timed step repeats its original transpose/preparation.
   srcR=Vector{Float64}(undef,3na*nb);srcN=Vector{Int32}(undef,na*nb)
   @inbounds for a in 0:na-1,b in 0:nb-1
    at=a+na*b;jt=(a*nb+b)*3
    for k in 0:2;srcR[jt+k+1]=s.rij[at+na*nb*k+1];end
    srcN[a*nb+b+1]=s.inside[at+1]
   end
   rij=similar(s.rij);inside=Vector{Float64}(undef,na*nb);wj=similar(s.w);rcut=similar(s.rcut);dedr=similar(s.rij);f=zeros(Float64,3nt);sumsq=0.0
   c=(;s,iu,parity,id,iz,icg,root,cg,zb,du,u,d,ut,y,r,rc,w)
   for step in 1:steps
    start=time_ns();fill!(f,0.0);jt=1;jjt=1
    @inbounds for a in 0:na-1,b in 0:nb-1
     at=a+na*b
     rij[at+1]=srcR[jt];jt+=1;rij[at+na*nb+1]=srcR[jt];jt+=1;rij[at+2na*nb+1]=srcR[jt];jt+=1
     inside[at+1]=Float64(srcN[jjt]);jjt+=1;wj[at+1]=1.0;rcut[at+1]=s.s[1]
    end
    copyto!(r.d,rij);copyto!(rc.d,rcut);copyto!(w.d,wj);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
    observe_input(ci,step,8,Array(r.d));observe_input(ci,step,9,Array(w.d));observe_input(ci,step,10,Array(rc.d));observe_input(ci,step,11,inside)
    start=time_ns();launch_pipeline!(c);copyto!(dedr,d.d)
    @inbounds for a in 0:na-1,b in 0:nb-1
     at=a+na*b;j=Int(inside[at+1])
     for k in 0:2;v=dedr[at+na*nb*k+1];f[a+nt*k+1]+=v;f[j+nt*k+1]-=v;end
    end
    @inbounds for a in 0:nt-1
     x=f[a+1]-s.reference[a+1];yy=f[a+nt+1]-s.reference[a+nt+1];z=f[a+2nt+1]-s.reference[a+2nt+1];sumsq+=x*x+yy*yy+z*z
    end
    seconds+=(time_ns()-start)*1e-9
    ccall(SNAP_OUTPUT,Cvoid,(Cint,Cint,Ptr{Float64},Ptr{Float64}),ci,step,dedr,f)
    for (bid,b) in enumerate(allbuffers);observe_guard(ci,step,bid-1,b);end
   end
   for (bid,b) in enumerate(static);observe_input(ci,steps+1,bid-1,Array(b.d));end
   ccall(SNAP_CASE_END,Cvoid,(Cint,Cdouble),ci,sumsq)
   for b in allbuffers;CUDA.unsafe_free!(b.d);CUDA.unsafe_free!(b.raw);end
  end
  trial>0&&ccall(SNAP_WRITE,Cvoid,(Cint,Cdouble),trial,seconds)
 end
 ccall(SNAP_FINISH,Cvoid,())
end
