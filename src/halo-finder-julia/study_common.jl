using CUDA,Libdl
CUDA.device!(0);CUDA.context()
struct HaloArgs
    count::Cint;count1::Cint;gx::Cint;gy::Cint;bx::Cint;by::Cint
    x::Ptr{Cvoid};y::Ptr{Cvoid};z::Ptr{Cvoid};mass::Ptr{Cvoid}
    x1::Ptr{Cvoid};y1::Ptr{Cvoid};z1::Ptr{Cvoid};mass1::Ptr{Cvoid}
    vx::Ptr{Cvoid};vy::Ptr{Cvoid};vz::Ptr{Cvoid}
    cut2::Float32;soft2::Float32;fcoeff::Float32;stream::Ptr{Cvoid}
end
const HALO_LIB=Libdl.dlopen(abspath(ARGS[1]));const HALO_RUN=Libdl.dlsym(HALO_LIB,:halo_run)
const HALO_ARGS_SIZE=Libdl.dlsym(HALO_LIB,:halo_argument_size)
const HALO_ROOT=abspath(ARGS[2])
function halo_wrap(p,n)
    @assert p!=C_NULL&&n>0
    unsafe_wrap(CuArray,CuPtr{Float32}(UInt(p)),(Int(n),);own=false)
end
function halo_callback(p::Ptr{HaloArgs})::Cint
    try
        a=unsafe_load(p)
        @assert a.bx==32&&a.by==4&&1<=a.count<=12&&a.count1 in (10001,31,65)
        names=(:x,:y,:z,:mass,:x1,:y1,:z1,:mass1,:vx,:vy,:vz)
        arrays=ntuple(i->halo_wrap(getproperty(a,names[i]),i in (5,6,7,8) ? a.count1 : a.count),11)
        c=NamedTuple{names}(arrays)
        GC.@preserve arrays begin
            launch_halo(a,c)
        end
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        return Cint(1)
    end
end
const HALO_CALLBACK=@cfunction(halo_callback,Cint,(Ptr{HaloArgs},))
function halo_study()
    @assert ccall(HALO_ARGS_SIZE,Csize_t,())==sizeof(HaloArgs)==136
    for trial in 0:5
        rc=ccall(HALO_RUN,Cint,(Ptr{Cvoid},Cint,Cstring),HALO_CALLBACK,trial,HALO_ROOT)
        @assert rc==0
    end
    nothing
end
