CUDA.device!(0);CUDA.context();CUDA.stream!(CUDA.legacy_stream())
struct PLaunch
    size::UInt32;len::UInt32;N::UInt32;blocks::UInt32;threads::UInt32;blen::UInt32
    mean::Float64;median::Float64;pointers::NTuple{6,Ptr{Cvoid}}
end
@assert sizeof(PLaunch)==88
const PERM_CALLBACKS=Ref(0);const PERM_KERNELS=Ref(0)
perm_wrap(p,::Type{T},n) where T=unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
function perm_callback(ptr::Ptr{PLaunch})::Cint
    try
        q=unsafe_load(ptr);n=Int(q.len);N=Int(q.N)
        D=perm_wrap(q.pointers[1],UInt8,n);S=perm_wrap(q.pointers[2],UInt8,n*N);B=perm_wrap(q.pointers[3],UInt8,q.size==1 ? Int(q.blen)*N : 1)
        R=perm_wrap(q.pointers[4],Float64,18);C=perm_wrap(q.pointers[5],UInt32,54)
        perm_launch(q,D,S,B,R,C);PERM_CALLBACKS[]+=1;PERM_KERNELS[]+=2
        Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);Cint(1)
    end
end
const PERM_CB=@cfunction(perm_callback,Cint,(Ptr{PLaunch},))
function perm_study()
    path=abspath(ARGS[1]);lib=Libdl.dlopen(path);entry=Libdl.dlsym(lib,:study_main)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(path)))," selected_native_kernel_fallback=false source_statistical_defects_preserved=true")
    status=ccall(entry,Cint,(Ptr{Cvoid},Cstring,Cstring,Cstring,Cint),PERM_CB,abspath(ARGS[2]),abspath(ARGS[3]),abspath(ARGS[4]),0)
    CUDA.device_synchronize();println("STUDY_AUTHORED_DEVICE_LAUNCHES ",PERM_KERNELS[]," full_batch_callbacks=",PERM_CALLBACKS[])
    status==0||error("Full source-behavior study rejected status=",status)
    PERM_CALLBACKS[]==18||error("Expected18 whole-batch callbacks from actual first-batch source earlystops")
    PERM_KERNELS[]==36||error("Expected36 genuine selected GPU launches including compilewarmup+five checked trials")
end
