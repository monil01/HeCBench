# Julia owns framework allocation, the complete repeated NCCL call loop and timing.
using CUDA
const CCL_LIB=joinpath(@__DIR__,"libstudy_nccl.so")
function ccl_ptr(a)
    Ptr{Cvoid}(UInt(pointer(a)))
end
function ccl_execute(allocate)
    localrank=ccall((:ccl_mpi_start,CCL_LIB),Cint,())
    CUDA.device!(Int(localrank))
    ccall((:ccl_comm_start,CCL_LIB),Cvoid,())
    for trial in 0:5
        ccall((:ccl_begin_trial,CCL_LIB),Cvoid,());elapsed=0.0
        for (c,n) in enumerate((17,1048576,10485760))
            send=Vector{Float32}(undef,n+2);recv=similar(send)
            ccall((:ccl_fill,CCL_LIB),Cvoid,(Ptr{Float32},Ptr{Float32},Cint,Cint),send,recv,n,c-1)
            ds=allocate(send);dr=allocate(recv)
            GC.@preserve ds dr begin
                s=ccl_ptr(ds);r=ccl_ptr(dr)
                sp=s+sizeof(Float32);rp=r+sizeof(Float32)
                CUDA.device_synchronize()
                ccall((:ccl_validate,CCL_LIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid}),sp,rp)
                if trial>0;ccall((:ccl_capture,CCL_LIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Cint,Cint,Cint),s,r,n,c-1,0);end
                ccall((:ccl_barrier,CCL_LIB),Cvoid,())
                start=ccall((:ccl_now,CCL_LIB),Cdouble,())
                for repeat in 1:50
                    ccall((:ccl_step,CCL_LIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Cint),sp,rp,n)
                end
                ccall((:ccl_wait,CCL_LIB),Cvoid,())
                dt=ccall((:ccl_now,CCL_LIB),Cdouble,())-start
                elapsed+=ccall((:ccl_report,CCL_LIB),Cdouble,(Cint,Cint,Cdouble),trial,c-1,dt/50)
                if trial>0;ccall((:ccl_capture,CCL_LIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Cint,Cint,Cint),s,r,n,c-1,1);end
            end
            CUDA.unsafe_free!(ds);CUDA.unsafe_free!(dr)
        end
        if trial>0;ccall((:ccl_write_trial,CCL_LIB),Cvoid,(Cint,Cdouble),trial,elapsed);end
    end
    ccall((:ccl_finish,CCL_LIB),Cvoid,())
end
