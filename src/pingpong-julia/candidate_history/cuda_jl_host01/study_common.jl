# Own Julia orchestration: no C host loop or source compute pipeline fallback.
CUDA.allowscalar(false)
const LIB=abspath(ARGS[1])
start()=ccall((:pp_start,LIB),Cint,())
guard()=ccall((:pp_guard,LIB),Cvoid,())
comm(s)=ccall((:pp_comm,LIB),Cvoid,(Ptr{Cvoid},),s)
recv(d,n,v)=ccall((:pp_recv,LIB),Cvoid,(CuPtr{Float64},Cint,Cint),pointer(d),n,v)
send(d,n,v)=ccall((:pp_send,LIB),Cvoid,(CuPtr{Float64},Cint,Cint),pointer(d),n,v)
contextwait()=ccall((:pp_context_wait,LIB),Cvoid,())
streamwait()=ccall((:pp_stream_wait,LIB),Cvoid,())
barrier()=ccall((:pp_barrier,LIB),Cvoid,())
now()=ccall((:pp_now,LIB),Cdouble,())
report(t,v,n,s)=ccall((:pp_report,LIB),Cdouble,(Cint,Cint,Cint,Cdouble),t,v,n,s)
begincapture()=ccall((:pp_begin,LIB),Cvoid,())
capture(h,n,stage)=GC.@preserve h ccall((:pp_capture,LIB),Cvoid,(Ptr{Float64},Cint,Cint),pointer(h),n,stage)
writecapture(t,s)=ccall((:pp_write,LIB),Cvoid,(Cint,Cdouble),t,s)
finish()=ccall((:pp_finish,LIB),Cvoid,())
function main()
 rank=Int(start());CUDA.device!(rank)
 stream=CUDA.stream();comm(convert(Ptr{Cvoid},getfield(stream,:handle)))
 for trial in 0:5
  guard();begincapture();elapsed=0.0
  for variant in 0:1,n in (65536,262144,1048576,257)
   guard();h=fill(-7777.25,n+64)
   for i in 0:n-1
    h[i+33]=rank==0 ? .5+(i%17-8)/32. : -3.+(i%13)/64.
   end
   raw=allocate_raw(h);d=unsafe_wrap(CuArray,CuPtr{Float64}(pointer(raw)+256),(n,);own=false)
   GC.@preserve raw d begin
    contextwait();copyto!(h,raw);trial>0 && capture(h,n,0)
    for _ in 1:5
     if rank==0
      send(d,n,variant);recv(d,n,variant)
     else
      recv(d,n,variant)
      if variant==0
       contextwait();increment!(d,n,variant);contextwait()
      else
       increment!(d,n,variant);streamwait()
      end
      send(d,n,variant)
     end
    end
    contextwait();copyto!(h,raw);trial>0 && capture(h,n,1)
    barrier();t=now()
    for _ in 1:50
     if rank==0
      send(d,n,variant);recv(d,n,variant)
     else
      recv(d,n,variant);send(d,n,variant)
     end
    end
    if variant==1;streamwait();else;contextwait();end
    seconds=now()-t;elapsed+=report(trial,variant,n,seconds)
    contextwait();copyto!(h,raw);trial>0 && capture(h,n,2)
   end
   CUDA.unsafe_free!(raw);guard()
  end
  trial>0 && writecapture(trial,elapsed);barrier()
 end
 finish()
end
main()
