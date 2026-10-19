using CUDA,Libdl
const MPI_LIBRARY=Libdl.dlopen(joinpath(pwd(),"studympi.so"))
const MS=Dict(s=>Libdl.dlsym(MPI_LIBRARY,Symbol("study_mpi_"*s)) for s in ["start","finish","barrier","guard","lengths","scatter","allgather","capture","max","abort"])
mpibarrier()=ccall(MS["barrier"],Cvoid,())
mpiguard()=ccall(MS["guard"],Cvoid,())
mpimax(t)=ccall(MS["max"],Cdouble,(Cdouble,),t)
ptr(a,i=1)=Ptr{Cvoid}(UInt(pointer(a,i)))
function capture(v::Vector{Float32})
 result=Vector{Float32}(undef,2length(v))
 ccall(MS["capture"],Cvoid,(Ptr{Float32},Cint,Ptr{Float32}),v,length(v),result)
 result
end
function ring(c,rank,reduce)
 half=cld(c.n,2);lens=(half,c.n-half);starts=(0,half)
 ccall(MS["lengths"],Cvoid,(Culong,),c.n)
 copyto!(c.output,c.input);CUDA.synchronize()
 recv=1-rank;send=rank
 GC.@preserve c begin
  ccall(MS["scatter"],Cvoid,(Ptr{Cvoid},Cint,Ptr{Cvoid},Cint),ptr(c.tmp),lens[recv+1],ptr(c.output,starts[send+1]+1),lens[send+1])
 end
 reduce(c,starts[recv+1],lens[recv+1]);CUDA.synchronize()
 send=1-rank;recv=rank
 GC.@preserve c begin
  ccall(MS["allgather"],Cvoid,(Ptr{Cvoid},Cint,Ptr{Cvoid},Cint),ptr(c.output,starts[recv+1]+1),lens[recv+1],ptr(c.output,starts[send+1]+1),lens[send+1])
 end
 CUDA.device_synchronize()
end
function study(alloc,reduce)
 rank=parse(Int,ENV["OMPI_COMM_WORLD_LOCAL_RANK"]);CUDA.device!(rank);stream=CuStream();CUDA.stream!(stream)
 ccall(MS["start"],Cint,())==rank||error("rankmapping")
 cs=[]
 for n in (1,32,33,256,4099,1048576)
  host=Float32[(rank+1)*.5f0+Float32(mod(i,17)-8)/32f0 for i in 0:n-1]
  ri=alloc(fill(-777f0,n+128));ro=alloc(fill(-777f0,n+128));rt=alloc(fill(-777f0,cld(n,2)+128))
  push!(cs,(n=n,host=host,ri=ri,ro=ro,rt=rt,input=view(ri,65:64+n),output=view(ro,65:64+n),tmp=view(rt,65:64+cld(n,2))))
 end
 function reset(c)
  fill!(c.ri,-777f0);copyto!(c.input,c.host);fill!(c.ro,-777f0);fill!(c.rt,-777f0);CUDA.device_synchronize()
 end
 for c in cs;reset(c);ring(c,rank,reduce);end
 # Warm timed launcher/clock-reduction wrapper outside trials too.
 mpimax(0.)
 check=Libdl.dlsym(MPI_LIBRARY,:study_check)
 for trial in 1:5
  mpiguard();out=Float32[6]
  for c in cs
   reset(c);both=capture(Array(c.input))
   if rank==0;push!(out,Float32(c.n));append!(out,both);end
  end
  seconds=0.
  for c in cs
   mpiguard();mpibarrier();start=time_ns()
   for r in 1:3;ring(c,rank,reduce);end
   elapsed=(time_ns()-start)*1e-9;seconds+=mpimax(elapsed)
   both=capture(Array(c.output));if rank==0;append!(out,both);end
   both=capture(Array(c.tmp));if rank==0;append!(out,both);end
   guards=Float32[]
   reinterpret(UInt32,Array(c.input))==reinterpret(UInt32,c.host)||error("readonlyinput")
   for raw in (c.ri,c.ro,c.rt);h=Array(raw);append!(guards,h[1:64]);append!(guards,h[end-63:end]);end
   both=capture(guards);if rank==0;append!(out,both);end
   mpiguard()
  end
  if rank==0
   path="trial$(trial).bin";open(io->write(io,out),path,"w");met=zeros(Float64,2)
   ccall(check,Cint,(Cstring,Ptr{Float64}),path,met)==0||error("independent/bothrankfullcheck")
   println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":",length(out),",\"max_abs_error\":",met[1],",\"max_rel_error\":",met[2],",\"timing_region\":\"sum6_max2rank_preallocated_source_ring_MPI_reduce_allgather_context_complete\",\"jit_excluded\":true,\"internal_repeats\":18,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  end
  mpibarrier()
 end
 ccall(MS["finish"],Cvoid,())
 # Framework allocations remain alive through MPI_Finalize, then normal CUDA finalization.
end
