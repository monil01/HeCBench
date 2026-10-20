using Libdl
const ROOT=normpath(@__DIR__)
const LIB=joinpath(pwd(),"study_backend.so")
const CASES=((4000,90000,10),(17,31,3),(2,1,2))
barrier()=ccall((:study_barrier,LIB),Cvoid,())
function guarded(rows,cols)
 ld=cld(rows,8)*8;ac=cld(cols,8)*8
 raw=CuArray{UInt8}(undef,ld*ac*4+512);fill!(raw,UInt8(0xa5));d=unsafe_wrap(CuArray,convert(CuPtr{Float32},pointer(raw)+256),ld*ac;own=false);fill!(d,0f0)
 (raw=raw,d=d,rows=rows,cols=cols,ld=ld,ac=ac)
end
function input!(a,ci,it,step,role,base)
 bytes=Array(a.raw)
 ccall((:study_input_frame,LIB),Cvoid,(Ptr{UInt8},Csize_t,Cint,Cint,Cint,Cint,UInt64,UInt64,UInt64,UInt64,UInt64),bytes,length(bytes),ci,it,step,role,a.rows,a.cols,a.ld,a.ac,base)
end
function output!(a,ci,it,step,bl,br,k)
 bytes=Array(a.raw)
 ccall((:study_output_frame,LIB),Cvoid,(Ptr{UInt8},Csize_t,Cint,Cint,Cint,UInt64,UInt64,UInt64,UInt64,UInt64,UInt64),bytes,length(bytes),ci,it,step,a.rows,a.ld,a.ac,bl,br,k)
 bytes
end
function guards!(a,ci,it,step,role)
 bytes=vcat(Array(view(a.raw,1:256)),Array(view(a.raw,length(a.raw)-255:length(a.raw))))
 ccall((:study_guard_frame,LIB),Cvoid,(Ptr{UInt8},Cint,Cint,Cint,Cint),bytes,ci,it,step,role)
end
function prep!(a,base,ci,s)
 ccall((:study_prep_begin,LIB),Cvoid,(Ptr{Cvoid},),s)
 launch!(a,UInt64(base))
 ccall((:study_prep_end,LIB),Cdouble,(Ptr{Cvoid},Cint),s,ci)
 nothing
end
function study()
 CUDA.allowscalar(false);CUDA.device!(0);CUDA.device_synchronize()
 println("SPARKLER_JULIA GPUcontextready rankenv=",get(ENV,"OMPI_COMM_WORLD_RANK","?"));flush(stdout)
 capture=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/sparkler_primary"))
 rank=Int(ccall((:study_start,LIB),Cint,(Cstring,Cstring),MODEL,capture))
 for trial in 0:5
  ccall((:study_begin_trial,LIB),Cvoid,(Cint,),trial)
  seconds=0.;hashes=zeros(UInt64,3);flops=zeros(Float64,3)
  for (cii,(nv,k,iterations)) in enumerate(CASES)
   ci=cii-1;base=nv*rank÷4;nlocal=nv*(rank+1)÷4-base;m=2cld(nv,4)
   println("SPARKLER_JULIA allocate rank=",rank," trial=",trial," case=",ci);flush(stdout)
   A=guarded(m,k);B=guarded(m,k);C=guarded(m,m)
   println("SPARKLER_JULIA allocated rank=",rank," case=",ci);flush(stdout)
   println("SPARKLER_JULIA getstream rank=",rank);flush(stdout)
   stream=CUDA.stream();println("SPARKLER_JULIA convertstream rank=",rank);flush(stdout)
   sp=convert(Ptr{Cvoid},getfield(stream,:handle))
   println("SPARKLER_JULIA beforehandle rank=",rank," stream=",sp);flush(stdout)
   h=ccall((:study_blas_start,LIB),Ptr{Cvoid},(Ptr{Cvoid},),sp)
   prep!(A,base,ci,sp);CUDA.device_synchronize();input!(A,ci,0,0,0,base)
   hash=UInt64(0);flop=0.
   for it in 1:iterations,step in 1:3
    CUDA.synchronize();barrier()
    right=(rank+step-1)%4;br=nv*right÷4;nlr=nv*(right+1)÷4-br;active=step<=(rank<2 ? 3 : 2)
    if active
     prep!(B,br,ci,sp)
     it==1&&input!(B,ci,it,step,1,br)
    end
    CUDA.synchronize();barrier();start=time_ns()
    if active
     GC.@preserve A B C begin
      ccall((:study_blas_step,LIB),Cvoid,(Ptr{Cvoid},Cint,Cint,Cint,Ptr{Float32},Cint,Ptr{Float32},Cint,Ptr{Float32},Cint),h,m,m,k,Ptr{Float32}(UInt(pointer(A.d))),A.ld,Ptr{Float32}(UInt(pointer(B.d))),B.ld,Ptr{Float32}(UInt(pointer(C.d))),C.ld)
     end
     flop+=2.0*m*m*k
    end
    CUDA.synchronize();CUDA.device_synchronize();barrier();seconds+=(time_ns()-start)*1e-9
    if active
     bytes=output!(C,ci,it,step,base,br,k);ch=reinterpret(Float32,bytes)
     guards!(A,ci,it,step,0);guards!(B,ci,it,step,1)
     for c in 0:nlr-1,r in 0:nlocal-1
      step==1&&r>=c&&continue
      value=UInt64(ch[65+r+C.ld*c]);rg=UInt64(r+base);cg=UInt64(c+br)
      hash+=UInt64(1)+((value*rg*cg)%UInt64(1<<40))
     end
    end
   end
   input!(A,ci,iterations+1,0,0,base)
   laststep=rank<2 ? 3 : 2;br=nv*((rank+laststep-1)%4)÷4
   input!(B,ci,iterations+1,laststep,1,br)
   hashes[cii]=ccall((:study_sum_u64,LIB),UInt64,(UInt64,),hash)
   flops[cii]=ccall((:study_sum_f64,LIB),Cdouble,(Cdouble,),flop)
   rank==0&&println("SPARKLER_CASE trial=",trial," case=",ci," hash=",hashes[cii]," flops=",flops[cii])
   ccall((:study_blas_finish,LIB),Cvoid,(Ptr{Cvoid},),h)
   for a in (A,B,C);CUDA.unsafe_free!(a.d);CUDA.unsafe_free!(a.raw);end
   GC.gc(false)
  end
  trial>0&&ccall((:study_write_trial,LIB),Cvoid,(Cint,Cdouble,Ptr{UInt64},Ptr{Float64}),trial,seconds,hashes,flops)
  barrier()
 end
 ccall((:study_finish,LIB),Cvoid,())
end
