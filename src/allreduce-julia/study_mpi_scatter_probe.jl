using CUDA,Libdl
const L=Libdl.dlopen(joinpath(pwd(),"studympi.so"))
const START=Libdl.dlsym(L,:study_mpi_start)
const EXCHANGE=Libdl.dlsym(L,:study_mpi_scatter)
const BARRIER=Libdl.dlsym(L,:study_mpi_barrier)
const FINISH=Libdl.dlsym(L,:study_mpi_finish)
const GUARD=Libdl.dlsym(L,:study_mpi_guard)
const ABORT=Libdl.dlsym(L,:study_mpi_abort)
function probe()
 rank=parse(Int,ENV["OMPI_COMM_WORLD_LOCAL_RANK"]);CUDA.device!(rank);stream=CuStream();CUDA.stream!(stream);ccall(START,Cint,())==rank||error("rank")
 n=524288;x=Float32[(rank+1)*.5f0+Float32(mod(i,17)-8)/32f0 for i in 0:n-1];expected=Float32[(2-rank)*.5f0+Float32(mod(i,17)-8)/32f0 for i in 0:n-1]
 rawin=CuArray(fill(-777f0,n+128));rawout=CuArray(fill(-777f0,n+128));copyto!(view(rawin,65:64+n),x);CUDA.device_synchronize()
 println("MPI_SCATTER_PROBE rank=",rank," device=",CUDA.device()," pool=",get(ENV,"JULIA_CUDA_MEMORY_POOL","unset")," input_ptr=",pointer(rawin,65)," output_ptr=",pointer(rawout,65))
 for iter in 1:20
  ccall(GUARD,Cvoid,());fill!(rawout,-777f0);CUDA.device_synchronize();ccall(BARRIER,Cvoid,())
  GC.@preserve rawin rawout begin
   ccall(EXCHANGE,Cvoid,(Ptr{Cvoid},Cint,Ptr{Cvoid},Cint),Ptr{Cvoid}(UInt(pointer(rawout,65))),n,Ptr{Cvoid}(UInt(pointer(rawin,65))),n)
  end
  CUDA.device_synchronize();h=Array(rawout);bad=count(i->h[i+64]!=expected[i],1:n);guards=all(==(-777f0),h[1:64])&&all(==(-777f0),h[end-63:end]);println("MPI_SCATTER_PROBE rank=",rank," iteration=",iter," wrong_values=",bad," guards=",guards)
  if bad>0||!guards
   open(io->write(io,h),"scatter_failed_rank$(rank).bin","w");ccall(ABORT,Cvoid,());error("full_device_exchange")
  end
 end
 ccall(FINISH,Cvoid,())
end
probe()
