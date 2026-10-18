# OpenMPI4.1.6 pinned host C ABI, same one-rank nonblocking selfexchange as native.
const MPI_LIB="/usr/lib/x86_64-linux-gnu/libmpi.so.40"
const MPI_WORLD=cglobal((:ompi_mpi_comm_world,MPI_LIB),Cvoid)
const MPI_DOUBLE=cglobal((:ompi_mpi_double,MPI_LIB),Cvoid)
const MPI_SUM=cglobal((:ompi_mpi_op_sum,MPI_LIB),Cvoid)
mpi_check(code)=code==0 ? nothing : error("MPI code $code")
function mpi_init()
 mpi_check(ccall((:MPI_Init,MPI_LIB),Cint,(Ptr{Cint},Ptr{Ptr{Ptr{UInt8}}}),C_NULL,C_NULL))
 size=Ref{Cint}(0);rank=Ref{Cint}(0)
 mpi_check(ccall((:MPI_Comm_size,MPI_LIB),Cint,(Ptr{Cvoid},Ref{Cint}),MPI_WORLD,size))
 mpi_check(ccall((:MPI_Comm_rank,MPI_LIB),Cint,(Ptr{Cvoid},Ref{Cint}),MPI_WORLD,rank))
 (size[]==1 && rank[]==0)||error("NamedsingleMPIrank required")
end
mpi_finalize()=mpi_check(ccall((:MPI_Finalize,MPI_LIB),Cint,()))
function mpi_irecv!(buf,tag,request,index)
 GC.@preserve buf request begin
  mpi_check(ccall((:MPI_Irecv,MPI_LIB),Cint,(Ptr{Cvoid},Cint,Ptr{Cvoid},Cint,Cint,Ptr{Cvoid},Ptr{Ptr{Cvoid}}),pointer(buf),length(buf),MPI_DOUBLE,0,tag,MPI_WORLD,pointer(request,index)))
 end
end
function mpi_isend!(buf,tag,request,index)
 GC.@preserve buf request begin
  mpi_check(ccall((:MPI_Isend,MPI_LIB),Cint,(Ptr{Cvoid},Cint,Ptr{Cvoid},Cint,Cint,Ptr{Cvoid},Ptr{Ptr{Cvoid}}),pointer(buf),length(buf),MPI_DOUBLE,0,tag,MPI_WORLD,pointer(request,index)))
 end
end
mpi_wait!(request)=mpi_check(ccall((:MPI_Waitall,MPI_LIB),Cint,(Cint,Ptr{Ptr{Cvoid}},Ptr{Cvoid}),length(request),request,C_NULL))
function mpi_sum(values)
 output=zeros(Float64,length(values))
 mpi_check(ccall((:MPI_Allreduce,MPI_LIB),Cint,(Ptr{Cvoid},Ptr{Cvoid},Cint,Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),values,output,length(values),MPI_DOUBLE,MPI_SUM,MPI_WORLD))
 return output
end
