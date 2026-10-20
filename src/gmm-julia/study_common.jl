using CUDA,Libdl
CUDA.device!(0);CUDA.context()
const GMM_LIB=Libdl.dlopen(abspath(ARGS[1]))
# Shared source CPU controller only; no native GPU entrypoint is looked up.
const GMM_RUN=Libdl.dlsym(GMM_LIB,:gmm_run)
const GMM_DATA=abspath(ARGS[2])
function gmm_wrap(ptr::Ptr{Cvoid},n)
    @assert ptr!=C_NULL && n>0
    unsafe_wrap(CuArray,CuPtr{Float32}(UInt(ptr)),(Int(n),);own=false)
end
function gmm_callback(stage::Cint,gx::Cint,gy::Cint,threads::Cint,data_ptr::Ptr{Cvoid},fields::Ptr{UInt64},d::Cint,m::Cint,n::Cint,capacity::Cint,like::Ptr{Cvoid})::Cint
    try
        @assert d==2&&n==992&&capacity==6&&1<=m<=capacity&&threads==256
        ptrs=ntuple(i->Ptr{Cvoid}(unsafe_load(fields,i)),8)
        sizes=(capacity,capacity,capacity,capacity,capacity*d,capacity*d*d,capacity*d*d,n*(capacity+6-rem(capacity,6)))
        arrays=ntuple(i->gmm_wrap(ptrs[i],sizes[i]),8)
        c=NamedTuple{(:N,:pi,:constant,:avgvar,:means,:R,:Rinv,:memberships)}(arrays)
        data=gmm_wrap(data_ptr,n*d)
        likelihood=like==C_NULL ? data : gmm_wrap(like,24)
        GC.@preserve arrays data likelihood begin
            launch_gmm(stage,gx,gy,threads,data,c,d,m,n,likelihood)
        end
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        return Cint(1)
    end
end
const GMM_CALLBACK=@cfunction(gmm_callback,Cint,(Cint,Cint,Cint,Cint,Ptr{Cvoid},Ptr{UInt64},Cint,Cint,Cint,Cint,Ptr{Cvoid}))
function gmm_study()
    @assert isfile(GMM_DATA)&&filesize(GMM_DATA)==33000
    # Complete source controller/all kernels warm up before all five trials.
    for trial in 0:5
        rc=ccall(GMM_RUN,Cint,(Ptr{Cvoid},Cint,Cstring,Cint,Cint),GMM_CALLBACK,trial,GMM_DATA,6,0)
        @assert rc==0
    end
    nothing
end
