using CUDA,Libdl
CUDA.device!(0);CUDA.context()
const LC_LIB=Libdl.dlopen(abspath(ARGS[1]));const LC_RUN=Libdl.dlsym(LC_LIB,:lanczos_run)
function lc_array(::Type{T},p::Ptr{Cvoid},n) where T
    @assert p!=C_NULL&&n>0
    unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(Int(n),);own=false)
end
function lanczos_callback(descriptor::Ptr{Cvoid})::Cint
    try
        # CPU sizeof/offsetof probe observed struct72B, scalar24, pointers32.
        ints=Ptr{Cint}(descriptor);stage=unsafe_load(ints,1);n=unsafe_load(ints,2)
        nz=unsafe_load(ints,3);g=unsafe_load(ints,4);blocks=unsafe_load(ints,5)
        scalar=unsafe_load(Ptr{Float64}(UInt(descriptor)+24))
        ptrs=Ptr{Ptr{Cvoid}}(UInt(descriptor)+32)
        count=stage==0 ? 3 : stage==1 ? 1 : stage==2 ? 2 : 5
        @assert n==257&&0<=stage<=3&&blocks>0
        arrays=ntuple(count) do k
            T=stage==3&&k<=2 ? Int32 : Float64
            len=stage==0&&k==3 ? blocks : stage==3&&k==1 ? n+1 : stage==3&&k in (2,3) ? nz : n
            lc_array(T,unsafe_load(ptrs,k),len)
        end
        GC.@preserve arrays begin
            launch_lanczos(stage,n,nz,g,blocks,scalar,arrays)
        end
        # Shared C dispatcher waits whole context before CPU observation/scalars.
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);return Cint(1)
    end
end
const LC_CALLBACK=@cfunction(lanczos_callback,Cint,(Ptr{Cvoid},))
function lanczos_study()
    for trial in 0:5
        path=trial==0 ? C_NULL : "output_trial$(trial).bin"
        @assert ccall(LC_RUN,Cint,(Cint,Ptr{Cvoid},Cstring),trial,LC_CALLBACK,path)==0
    end
end
