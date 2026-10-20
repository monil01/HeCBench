CUDA.device!(0);CUDA.context();CUDA.stream!(CUDA.legacy_stream())
struct NBLaunch
    nAtoms::Int32;nCJ::Int32;nSCI::Int32;nEX::Int32
    numTypes::Int32;gridZ::Int32;repeat::Int32;calcShift::Int32
    cutoffSq::Float32;beta::Float32;epsFac::Float32;padding::Float32
    pointers::NTuple{9,Ptr{Cvoid}}
end
@assert sizeof(NBLaunch)==120
const NB_KERNEL_LAUNCHES=Ref(0);const NB_PHASE_CALLBACKS=Ref(0)
@inline nb_wrap(p::Ptr{Cvoid},::Type{T},n) where T=unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
function nb_callback(p::Ptr{NBLaunch})::Cint
    try
        q=unsafe_load(p);n=q.nAtoms
        a=(nb_wrap(q.pointers[1],Float32,4*n),nb_wrap(q.pointers[2],Float32,3*n),nb_wrap(q.pointers[3],Float32,135),nb_wrap(q.pointers[4],Float32,135),nb_wrap(q.pointers[5],UInt32,8*q.nCJ),nb_wrap(q.pointers[6],Int32,4*q.nSCI),nb_wrap(q.pointers[7],UInt32,32*q.nEX),nb_wrap(q.pointers[8],Int32,n),nb_wrap(q.pointers[9],Float32,2*q.numTypes*q.numTypes))
        nb_launch_phase(q,a)
        NB_KERNEL_LAUNCHES[]+=Int(q.repeat);NB_PHASE_CALLBACKS[]+=1
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        return Cint(1)
    end
end
const NB_CALLBACK=@cfunction(nb_callback,Cint,(Ptr{NBLaunch},))
function nb_study()
    path=abspath(ARGS[1]);lib=Libdl.dlopen(path);entry=Libdl.dlsym(lib,:nb_run)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(path)))," native_managed_allocation_control=true selected_native_kernel_fallback=false")
    status=ccall(entry,Cint,(Cstring,Cstring,Ptr{Cvoid},Cint),abspath(ARGS[2]),abspath(ARGS[3]),NB_CALLBACK,0)
    CUDA.device_synchronize()
    println("STUDY_ACTUAL_AUTHORED_KERNEL_LAUNCHES ",NB_KERNEL_LAUNCHES[]," phase_callbacks=",NB_PHASE_CALLBACKS[])
    status==0||error("Nbnxmfullstudyrejectedstatus=",status)
    NB_KERNEL_LAUNCHES[]==120030||error("Unexpectedsource-shapedlaunchcount")
    NB_PHASE_CALLBACKS[]==42||error("Expected36studyphases+6sourceuntimedwarmupcallbacks")
end
