using CUDA
using Libdl
const STUDY_CUBLAS_LIBRARY="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib/libcublas.so.12.9.0.13"
function study_prepare(a,b,c,m,oa,ob,oc,k,n)
    lib=Libdl.dlopen(STUDY_CUBLAS_LIBRARY);create=Libdl.dlsym(lib,:cublasCreate_v2);destroy=Libdl.dlsym(lib,:cublasDestroy_v2);call=Libdl.dlsym(lib,:cublasGemmGroupedBatchedEx);handle=Ref{Ptr{Cvoid}}();status=ccall(create,Cint,(Ref{Ptr{Cvoid}},),handle);status==0||error("cublasCreate status",status)
    trans=zeros(Int32,length(m));ns=fill(Int32(n),length(m));ks=fill(Int32(k),length(m));sizes=ones(Int32,length(m));alpha=ones(Float32,length(m));beta=zeros(Float32,length(m));ap=CuArray([pointer(a,Int(x)+1) for x in oa[1:end-1]]);bp=CuArray([pointer(b,Int(x)+1) for x in ob[1:end-1]]);cp=CuArray([pointer(c,Int(x)+1) for x in oc[1:end-1]])
    CUDA.synchronize()
    launch! = function()
        status=ccall(call,Cint,(Ptr{Cvoid},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Float32},CUDA.CuPtr{Cvoid},Cint,Ptr{Int32},CUDA.CuPtr{Cvoid},Cint,Ptr{Int32},Ptr{Float32},CUDA.CuPtr{Cvoid},Cint,Ptr{Int32},Cint,Ptr{Int32},Cint),handle[],trans,trans,m,ns,ks,alpha,ap,2,m,bp,2,ks,beta,cp,2,m,length(m),sizes,68)
        status==0||error("cublasGemmGroupedBatchedEx status",status)
    end
    finalizer(c) do _
        ccall(destroy,Cint,(Ptr{Cvoid},),handle[])
    end
    launch!
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
