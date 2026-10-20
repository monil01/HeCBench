CUDA.device!(0)
CUDA.context()
const MM_STREAM=CUDA.CuStream()
CUDA.stream!(MM_STREAM)
struct MMCall
    op::Int32;grid::Int32;block::Int32;ndims::Int32
    nfibers::UInt64;rank::UInt64;wps::UInt64
    logWPC::Int32;fpw::Int32;logFPW::Int32;padding::Int32
    mode::UInt64
    pointers::NTuple{12,Ptr{Cvoid}}
    lengths::NTuple{12,UInt64}
end
@assert sizeof(MMCall)==256
const MM_COUNTS=zeros(Int,7)
const MM_EMPTY=(CuArray{UInt64}(undef,0),CuArray{Float32}(undef,0),CuArray{Float64}(undef,0))
function mm_wrap(::Type{T},p,n) where T
    if p==C_NULL
        n==0||error("Nonnull length for null source field")
        return MM_EMPTY[T==UInt64 ? 1 : T==Float32 ? 2 : 3]
    end
    unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
end
function mm_callback(ptr::Ptr{MMCall})::Cint
    try
        q=unsafe_load(ptr)
        arrays=ntuple(12) do j
            mm_wrap(j==9 ? Float64 : j==1||j>=10 ? Float32 : UInt64,q.pointers[j],q.lengths[j])
        end
        GC.@preserve arrays begin
            mm_launch(q,arrays)
        end
        MM_COUNTS[Int(q.op)+1]+=1
        Cint(0)
    catch e
        showerror(stderr,e,catch_backtrace());println(stderr);flush(stderr);Cint(1)
    end
end
const MM_CALLBACK=@cfunction(mm_callback,Cint,(Ptr{MMCall},))
function mm_study()
    length(ARGS)==4||error("Usage: main.jl LIB FIXTURES CANONICAL PRIVATE_ROOT")
    library=abspath(ARGS[1]);fixtures=abspath(ARGS[2]);canonical=abspath(ARGS[3]);private=abspath(ARGS[4]);mkpath(private)
    lib=Libdl.dlopen(library);entry=Libdl.dlsym(lib,:mm_run)
    println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(library)))," native_CPU_controller=true native_GPU_fallback=false owned_serial_stream=true")
    path,fd=mktemp(private;cleanup=false)
    Base.Libc.flush_cstdio()
    status=redirect_stdout(fd) do
        value=ccall(entry,Cint,(Cstring,Cstring,Cstring,Ptr{Cvoid},Cint),fixtures,canonical,private,MM_CALLBACK,0)
        Base.Libc.flush_cstdio();value
    end
    close(fd);print(read(path,String));flush(stdout)
    CUDA.device_synchronize(;blocking=true)
    println("STUDY_C_STDOUT_CAPTURE ",path," sha256=",bytes2hex(sha256(read(path))))
    println("STUDY_FRAMEWORK_DEVICE_DISPATCHES ",sum(MM_COUNTS)," per_entry=",MM_COUNTS)
    status==0||error("Full MMCSF strict accuracy gate rejected status=",status)
    MM_COUNTS==fill(12,7)||error("Allsevenbranch/warmup+five dispatch count mismatch")
end
