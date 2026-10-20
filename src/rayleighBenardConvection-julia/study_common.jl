CUDA.device!(0);CUDA.context();CUDA.stream!(CUDA.legacy_stream())
struct RBCall
    op::Cint;n::Cint;gx::Cint;gy::Cint;threads::Cint;iu::Cint;iv::Cint;pad::Cint
    pointers::NTuple{3,Ptr{Cvoid}}
end
@assert sizeof(RBCall)==56
const RBC_DX=1f0/(500f0-1f0)
const RBC_CAP=(RBC_DX*RBC_DX)/4f0
const RBC_COUNTS=zeros(Int,7)
rbc_wrap(p,n)=unsafe_wrap(CuArray,CuPtr{Float32}(UInt(p)),(Int(n),);own=false)
function rbc_callback(p::Ptr{RBCall})::Cint
    try
        q=unsafe_load(p);len=q.op==6 ? 498000 : Int(q.n)
        A=rbc_wrap(q.pointers[1],len)
        B=q.pointers[2]==C_NULL ? A : rbc_wrap(q.pointers[2],len)
        out=q.pointers[3]==C_NULL ? A : rbc_wrap(q.pointers[3],1)
        rbc_launch(q,A,B,out);RBC_COUNTS[Int(q.op)+1]+=1
        Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);Cint(1)
    end
end
const RBC_CB=@cfunction(rbc_callback,Cint,(Ptr{RBCall},))
function rbc_study()
    library=abspath(ARGS[1]);lib=Libdl.dlopen(library);fn=Libdl.dlsym(lib,:study_main)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(library)))," native_selected_GPU_fallback=false original_cuBLAS_host_controller_retained=true")
    status=ccall(fn,Cint,(Ptr{Cvoid},Cstring,Cint),RBC_CB,abspath(ARGS[2]),0)
    CUDA.device_synchronize()
    println("STUDY_AUTHORED_DEVICE_DISPATCHES ",sum(RBC_COUNTS)," per_entry=",RBC_COUNTS)
    status==0||error("Fullsource1000stepRBC rejected status=",status)
    # Three heat samples per full source run; six executions include warmup+five.
    RBC_COUNTS==6 .* [3000,6000,6,3024,3006,0,1000]||error("Actual fullpipeline dispatch count differs from declared source")
end
