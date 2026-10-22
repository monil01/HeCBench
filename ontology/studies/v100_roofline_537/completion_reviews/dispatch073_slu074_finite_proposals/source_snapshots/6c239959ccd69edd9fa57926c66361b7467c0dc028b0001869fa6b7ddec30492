CUDA.device!(0);CUDA.context();CUDA.stream!(CUDA.legacy_stream())
struct SLUCall
    op::Cint;threads::Cint;blocks::Cint;smem::Cint
    n::UInt32;nnz::UInt32;col::UInt32
    head::Cint;pos::Cint;slot::Cint
    pert::Float32;pad::Cint
    pointers::NTuple{9,Ptr{Cvoid}}
end
@assert sizeof(SLUCall)==120
const SLU_COUNTS=zeros(Int,6)
slu_wrap(::Type{T},p,n) where T=unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
function slu_callback(p::Ptr{SLUCall})::Cint
    try
        q=unsafe_load(p);n=Int(q.n);z=Int(q.nnz)
        a=slu_wrap(UInt32,q.pointers[1],n+1);r=slu_wrap(UInt32,q.pointers[2],z)
        v=slu_wrap(Float32,q.pointers[3],z);d=slu_wrap(UInt32,q.pointers[4],n)
        rp=slu_wrap(UInt32,q.pointers[5],n+1);ci=slu_wrap(UInt32,q.pointers[6],z)
        rd=slu_wrap(UInt32,q.pointers[7],n);lev=slu_wrap(Int32,q.pointers[8],n)
        tmp=slu_wrap(Float32,q.pointers[9],10353*n)
        slu_launch(q,a,r,v,d,rp,ci,rd,lev,tmp)
        SLU_COUNTS[Int(q.op)+1]+=1;Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);Cint(1)
    end
end
const SLU_CALLBACK=@cfunction(slu_callback,Cint,(Ptr{SLUCall},))
function slu_study()
    libpath=abspath(ARGS[1]);lib=Libdl.dlopen(libpath);fn=Libdl.dlsym(lib,:study_main)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(libpath)))," native_selected_GPU_fallback=false CPU_NICSLU_symbolic_solve_retained=true CUDA_legacy_stream=true")
    result=ccall(fn,Cint,(Ptr{Cvoid},Cstring,Cstring,Cint),SLU_CALLBACK,abspath(ARGS[2]),abspath(ARGS[3]),0)
    CUDA.device_synchronize()
    println("STUDY_AUTHORED_DEVICE_DISPATCHES ",sum(SLU_COUNTS)," per_entry=",SLU_COUNTS)
    result==0||error("Full SLU independent gate failed status=",result)
    all(>(0),SLU_COUNTS)||error("Allsixselectedsourceentryspecializationsmustexecute")
end
