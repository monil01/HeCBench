# Same retained scientific CPU host CABI, authored target GPU callbacks only.
using Libdl
const HANDLES=Dict{Tuple{UInt,Int},CuArray{Int16,1}}()
const STREAMS=CuStream[]
function wrap_buffer(p::Ptr{Cvoid},n::Int)
    get!(HANDLES,(UInt(p),n)) do
        unsafe_wrap(CuArray,CuPtr{Int16}(p),(n,);own=false)
    end
end
function callback(op::Cint,tasks::Cint,shmem::Cint,stream::Ptr{Cvoid},del::Ptr{Cvoid},add::Ptr{Cvoid},out::Ptr{Cvoid},nd::Csize_t,na::Csize_t)::Cint
    try
        actual=Ptr{Cvoid}(STREAMS[Int(op)+1].handle)
        actual==stream || error("Source/framework stream ownership mismatch")
        D=wrap_buffer(del,5003);O=wrap_buffer(out,1225)
        if op==0
            launch_delete(D,O,Int(tasks),Int(shmem),STREAMS[1])
        else
            A=wrap_buffer(add,7653)
            launch_add(D,A,O,Int(tasks),Int(shmem),STREAMS[2])
        end
        return Cint(0)
    catch e
        showerror(stderr,e,catch_backtrace());println(stderr);return Cint(1)
    end
end
function main()
    length(ARGS)==4||error("usage: target LIB DATA OUTPUT trials")
    CUDA.device!(0);CUDA.functional(true)||error("CUDA not functional")
    append!(STREAMS,[CuStream(flags=CUDA.STREAM_DEFAULT),CuStream(flags=CUDA.STREAM_DEFAULT)])
    lib=Libdl.dlopen(ARGS[1],Libdl.RTLD_LOCAL|Libdl.RTLD_DEEPBIND)
    register=Libdl.dlsym(lib,:sss_register_streams);run=Libdl.dlsym(lib,:sss_run)
    ccall(register,Cvoid,(Ptr{Cvoid},Ptr{Cvoid}),Ptr{Cvoid}(STREAMS[1].handle),Ptr{Cvoid}(STREAMS[2].handle))
    cb=@cfunction(callback,Cint,(Cint,Cint,Cint,Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Csize_t,Csize_t))
    # Entire first full source search is warmup: both entries compiled before trials1-5.
    status=ccall(run,Cint,(Cstring,Cstring,Ptr{Cvoid},Cint),ARGS[2],ARGS[3],cb,parse(Cint,ARGS[4]))
    CUDA.device_synchronize(blocking=true);status==0||error("Source scientific controller/independent checks failed")
    empty!(HANDLES)
end
