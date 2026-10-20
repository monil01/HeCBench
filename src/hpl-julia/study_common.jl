using CUDA,Libdl
CUDA.device!(0);CUDA.context()
const HPL_LIB=Libdl.dlopen(abspath(ARGS[1]))
const HPL_INIT=Libdl.dlsym(HPL_LIB,:hpl_initialize)
const HPL_RUN=Libdl.dlsym(HPL_LIB,:hpl_run)
const HPL_FINISH=Libdl.dlsym(HPL_LIB,:hpl_finalize)
const HPL_OBSERVE=Libdl.dlsym(HPL_LIB,:hpl_observe)
# The original C/MPI HPL solver/controller is shared. Only framework-owned
# allocation/copy, modern cuBLAS orchestration and completion occur here.
# There is no authored scientific CUDA kernel in the original benchmark.
function hpl_callback(kind::Cint,m::Cint,n::Cint,k::Cint,alpha::Cdouble,A::Ptr{Cdouble},lda::Cint,B::Ptr{Cdouble},ldb::Cint,beta::Cdouble,C::Ptr{Cdouble},ldc::Cint)::Cvoid
    try
        @assert kind in (0,1) && m>0 && n>0 && lda>=m
        hA=unsafe_wrap(Array,A,(Int(lda),Int(kind==0 ? k : m));own=false)
        hB=unsafe_wrap(Array,B,(Int(ldb),Int(n));own=false)
        hC=kind==0 ? unsafe_wrap(Array,C,(Int(ldc),Int(n));own=false) : nothing
        dA=framework_array(hA);dB=framework_array(hB)
        dC=kind==0 ? framework_array(hC) : nothing
        GC.@preserve hA hB hC dA dB dC begin
            CUDA.device_synchronize()
            pA=Ptr{Float64}(UInt(pointer(dA)));pB=Ptr{Float64}(UInt(pointer(dB)))
            pC=kind==0 ? Ptr{Float64}(UInt(pointer(dC))) : Ptr{Float64}(C_NULL)
            ccall(HPL_OBSERVE,Cvoid,(Cint,Cint,Cint,Cint,Cdouble,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Cdouble,Ptr{Cdouble},Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),kind,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,pA,pB,pC)
            if kind==0
                CUDA.CUBLAS.gemm!('N','N',alpha,view(dA,1:Int(m),1:Int(k)),view(dB,1:Int(k),1:Int(n)),beta,view(dC,1:Int(m),1:Int(n)))
            else
                CUDA.CUBLAS.trsm!('L','L','N','U',alpha,view(dA,1:Int(m),1:Int(m)),view(dB,1:Int(m),1:Int(n)))
            end
            CUDA.device_synchronize()
            if kind==0;copyto!(hC,dC);else;copyto!(hB,dB);end
            CUDA.device_synchronize()
        end
        CUDA.unsafe_free!(dA);CUDA.unsafe_free!(dB)
        if kind==0;CUDA.unsafe_free!(dC);end
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        # A failed callback cannot be allowed to continue the native solver.
        Base.exit(3)
    end
    nothing
end
const HPL_CALLBACK=@cfunction(hpl_callback,Cvoid,(Cint,Cint,Cint,Cint,Cdouble,Ptr{Cdouble},Cint,Ptr{Cdouble},Cint,Cdouble,Ptr{Cdouble},Cint))
function hpl_study()
    @assert ccall(HPL_INIT,Cint,())==0
    println("HPL_JULIA_LIBRARY_PATHS ",join(filter(p->occursin("cublas",p)||occursin("openblas",p),Libdl.dllist()),";"))
    for trial in 0:5
        path=trial==0 ? C_NULL : "output_trial$(trial).bin"
        @assert ccall(HPL_RUN,Cint,(Cint,Ptr{Cvoid},Cstring),trial,HPL_CALLBACK,path)==0
    end
    @assert ccall(HPL_FINISH,Cvoid,())===nothing
end
