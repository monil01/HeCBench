using CUDA,Libdl
include("c_memory.jl");include("strash_helpers.jl");include("generated_strash.jl")
CUDA.device!(0);CUDA.context()
const LRLIB=Libdl.dlopen(abspath(ARGS[1]));const LR_RUN=Libdl.dlsym(LRLIB,:lr_strash_unit)
const LR_COUNTS=Dict{String,Int}()
Base.@noinline function lr_launch_device(f,args,threads,grid)
    CUDA.@cuda threads=threads blocks=grid stream=CUDA.legacy_stream() maxregs=128 f(nothing,args...)
    return nothing
end
function lr_callback(d::Ptr{Cvoid})::Cint
 try
    name=unsafe_string(unsafe_load(Ptr{Cstring}(d)));words=Ptr{UInt64}(d)
    grid=ntuple(i->Int(unsafe_load(words,1+i)),3);threads=ntuple(i->Int(unsafe_load(words,4+i)),3);n=Int(unsafe_load(words,8));@assert n<=32
    v=ntuple(i->unsafe_load(words,40+i),n);i(k)=lr_cast(Int32,v[k]);p(k,T)=lr_ptr(T,v[k])
        if name=="strash.buildHashTable"
            @assert n==7;f=lr_buildHashTable;args=(p(1,Int32),p(2,Int32),p(3,UInt64),p(4,UInt32),i(5),i(6),i(7),)
        elseif name=="strash.markReadyNodes"
            @assert n==6;f=lr_markReadyNodes;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),i(6),)
        elseif name=="strash.insertLevelNodes"
            @assert n==10;f=lr_insertLevelNodes;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,UInt64),p(6,UInt32),i(7),p(8,UInt64),i(9),i(10),)
        elseif name=="strash.updateLevelNodesNewIds"
            @assert n==7;f=lr_updateLevelNodesNewIds;args=(p(1,Int32),p(2,Int32),p(3,UInt64),p(4,UInt64),p(5,UInt32),i(6),i(7),)
        elseif name=="strash.assignOld2NewConsecutiveIds"
            @assert n==4;f=lr_assignOld2NewConsecutiveIds;args=(p(1,UInt32),p(2,Int32),i(3),i(4),)
        elseif name=="strash.unbindKeysReId"
            @assert n==6;f=lr_unbindKeysReId;args=(p(1,UInt64),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),i(6),)
        elseif name=="strash.poReId"
            @assert n==4;f=lr_poReId;args=(p(1,Int32),p(2,Int32),p(3,Int32),i(4),)
        elseif name=="strash.poUpdateId"
            @assert n==4;f=lr_poUpdateId;args=(p(1,Int32),p(2,Int32),p(3,Int32),i(4),)
        elseif name=="strash.markDanglingNodesIter"
            @assert n==6;f=lr_markDanglingNodesIter;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),i(5),i(6),)
        elseif name=="hash64_32.getEntryIndicator"
            @assert n==3;f=lr_s_getEntryIndicator;args=(p(1,Int32),p(2,UInt64),i(3))
        elseif name=="hash64_32.gatherKeyValues"
            @assert n==6;f=lr_s_gatherKeyValues;args=(p(1,Int32),p(2,UInt64),p(3,UInt32),p(4,UInt64),p(5,UInt32),i(6))
        else
            error("No Julia implementation for interceptedsourcekernel ",name)
        end
    Base.invokelatest(lr_launch_device,f,args,threads,grid)
    LR_COUNTS[name]=get(LR_COUNTS,name,0)+1
    return Cint(0)
 catch err
    showerror(stderr,err,catch_backtrace());println(stderr);return Cint(1)
 end
end
precompile(lr_callback,(Ptr{Cvoid},));const LR_CB=@cfunction(lr_callback,Cint,(Ptr{Cvoid},))
@assert ccall(LR_RUN,Cint,(Cstring,Cstring,Ptr{Cvoid}),ARGS[2],ARGS[3]*".native",C_NULL)==0
@assert ccall(LR_RUN,Cint,(Cstring,Cstring,Ptr{Cvoid}),ARGS[2],ARGS[3]*".julia",LR_CB)==0
CUDA.device_synchronize()
a=read(ARGS[3]*".native");b=read(ARGS[3]*".julia");@assert a==b
println("UNIT_SOURCE_FULL_STRASH_FFI_PASS rawbytes=",length(a)," dispatch=",LR_COUNTS," nativekernel_fallback=false fullresyn2pending=true")
