CUDA.device!(0)
CUDA.context()
const XL_STREAM=CUDA.CuStream()
CUDA.stream!(XL_STREAM)
struct XLCall
    op::Int32;n::Int32
    pbf::Ptr{Float64};map::Ptr{Int32};D::Ptr{Float64};out::Ptr{Float64};Q::Ptr{Float64}
end
@assert sizeof(XLCall)==48
const XL_COUNTS=zeros(Int,4)
xl_wrap(::Type{T},p,n) where T=unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
function xl_callback(ptr::Ptr{XLCall})::Cint
    try
        q=unsafe_load(ptr);n=Int(q.n)
        n==21||error("Named full H2O primitive count changed")
        pbf=xl_wrap(Float64,q.pbf,8*n);mapping=xl_wrap(Int32,q.map,n)
        D=xl_wrap(Float64,q.D,28);out=xl_wrap(Float64,q.out,n*(n+1)÷2);Q=xl_wrap(Float64,q.Q,28)
        GC.@preserve pbf mapping D out Q begin
            xl_launch(q.op,pbf,mapping,q.n,D,out,Q)
        end
        XL_COUNTS[Int(q.op)+1]+=1
        Cint(0)
    catch e
        showerror(stderr,e,catch_backtrace());println(stderr);flush(stderr)
        Cint(1)
    end
end
const XL_CALLBACK=@cfunction(xl_callback,Cint,(Ptr{XLCall},))
function xl_study()
    length(ARGS)==4||error("Usage: main.jl STUDY_LIBRARY MODE FIXTURES PRIVATE_ROOT")
    library=abspath(ARGS[1]);mode=parse(Cint,ARGS[2]);mode in (0,1)||error("Precision mode")
    fixtures=abspath(ARGS[3]);private=abspath(ARGS[4]);lib=Libdl.dlopen(library);entry=Libdl.dlsym(lib,:xl_study_main)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(library)))," source_CPU_GSL_full_SCF_controller=true selected_native_GPU_fallback=false actual_framework_callback=true owned_serial_stream=true")
    # Julia stdout pipes may be nonblocking; unmediated C++ iostream writes can
    # truncate a >64KiB structured row. A regular file preserves the original
    # bytes; Julia re-emits them after completion, entirely outside stage ROIs.
    mkpath(private)
    c_stdout_path,c_stdout=mktemp(private;cleanup=false)
    println("STUDY_C_STDOUT_FD_FLAGS ",ccall(:fcntl,Cint,(Cint,Cint),1,3))
    Base.Libc.flush_cstdio()
    status=redirect_stdout(c_stdout) do
        value=ccall(entry,Cint,(Ptr{Cvoid},Cint,Cstring,Cstring,Cint),XL_CALLBACK,mode,fixtures,private,0)
        Base.Libc.flush_cstdio()
        value
    end
    close(c_stdout)
    print(read(c_stdout_path,String));flush(stdout)
    println("STUDY_C_STDOUT_CAPTURE ",c_stdout_path," sha256=",bytes2hex(sha256(read(c_stdout_path))))
    CUDA.device_synchronize(;blocking=true)
    expected=mode==0 ? [54,54,0,0] : [0,0,54,54]
    println("STUDY_FRAMEWORK_DEVICE_DISPATCHES ",sum(XL_COUNTS)," per_entry=",XL_COUNTS)
    status==0||error("Full SCF scientific/native-reference gate rejected status=",status)
    XL_COUNTS==expected||error("Full warmup+five complete SCF dispatch coverage differs")
end
