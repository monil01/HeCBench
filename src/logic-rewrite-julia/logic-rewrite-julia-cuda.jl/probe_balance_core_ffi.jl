using CUDA,Libdl
include("c_memory.jl");include("strash_helpers.jl");include("balance_helpers.jl");include("generated_balance.jl")
CUDA.device!(0);CUDA.context()
const LRLIB=Libdl.dlopen(abspath(ARGS[1]));const LR_RUN=Libdl.dlsym(LRLIB,:lr_balance_unit)
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
        if name=="balance.checkCoverTravEnd"
            @assert n==6;f=lr_checkCoverTravEnd;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),i(5),i(6),)
        elseif name=="balance.prepareDataToInsert"
            @assert n==18;f=lr_prepareDataToInsert;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt64),p(5,UInt32),p(6,Int32),p(7,UInt32),p(8,UInt64),p(9,UInt32),i(10),p(11,UInt32),p(12,UInt32),i(13),i(14),i(15),i(16),i(17),i(18),)
        elseif name=="balance.sharedNodeDrivenPermute"
            @assert n==8;f=lr_sharedNodeDrivenPermute;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt64),p(5,UInt32),i(6),i(7),i(8),)
        elseif name=="balance.prepareReconstructArrays"
            @assert n==17;f=lr_prepareReconstructArrays;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,UInt32),p(9,UInt32),i(10),p(11,Int32),p(12,Int32),p(13,Int32),i(14),i(15),i(16),i(17),)
        elseif name=="balance.addBackLocalArrays"
            @assert n==8;f=lr_addBackLocalArrays;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt32),p(5,Int32),p(6,UInt32),i(7),i(8),)
        elseif name=="balance.recordReconstructedCovers"
            @assert n==7;f=lr_recordReconstructedCovers;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),i(6),i(7),)
        elseif name=="balance.genReadyMask"
            @assert n==9;f=lr_genReadyMask;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),i(8),i(9),)
        elseif name=="balance.gatherByScannedMask"
            @assert n==4;f=lr_gatherByScannedMask;args=(p(1,Int32),p(2,Int32),i(3),i(4),)
        elseif name=="balance.markIsCoverOutput"
            @assert n==3;f=lr_markIsCoverOutput;args=(p(1,Int32),p(2,Int32),i(3),)
        elseif name=="balance.getCoverToNodeIdMapping"
            @assert n==4;f=lr_getCoverToNodeIdMapping;args=(p(1,Int32),p(2,Int32),p(3,Int32),i(4),)
        elseif name=="balance.gatherWithFilter"
            @assert n==10;f=lr_gatherWithFilter;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,Int32),i(9),i(10),)
        elseif name=="balance.coverFindingToTable"
            @assert n==13;f=lr_coverFindingToTable;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,Int32),p(9,Int32),p(10,Int32),i(11),i(12),i(13),)
        elseif name=="balance.findLevelNodeRanges"
            @assert n==3;f=lr_findLevelNodeRanges;args=(p(1,Int32),p(2,Int32),i(3),)
        elseif name=="balance.parseOutputRes"
            @assert n==9;f=lr_parseOutputRes;args=(p(1,UInt64),p(2,UInt32),p(3,UInt32),i(4),p(5,Int32),p(6,Int32),p(7,Int32),i(8),i(9),)
        elseif name=="balance.processPO"
            @assert n==10;f=lr_processPO;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt32),p(5,UInt32),i(6),p(7,Int32),p(8,Int32),i(9),i(10),)
        elseif name=="hash64_32.getEntryIndicator"
            @assert n==3;f=lr_s_getEntryIndicator;args=(p(1,Int32),p(2,UInt64),i(3),)
        elseif name=="hash64_32.gatherKeyValues"
            @assert n==6;f=lr_s_gatherKeyValues;args=(p(1,Int32),p(2,UInt64),p(3,UInt32),p(4,UInt64),p(5,UInt32),i(6),)
        elseif name=="hash64_32.insert_batch_no_update_kernel"
            @assert n==6;f=lr_s_insert_batch_no_update_kernel;args=(p(1,UInt64),p(2,UInt32),p(3,UInt64),p(4,UInt32),i(5),i(6),)
        elseif name=="hash64_32.insert_batch_no_update_masked_kernel"
            @assert n==7;f=lr_s_insert_batch_no_update_masked_kernel;args=(p(1,UInt64),p(2,UInt32),p(3,UInt64),p(4,UInt32),p(5,Int32),i(6),i(7),)
        elseif name=="hash64_32.retrieve_batch_kernel"
            @assert n==6;f=lr_s_retrieve_batch_kernel;args=(p(1,UInt64),p(2,UInt32),p(3,UInt64),p(4,UInt32),i(5),i(6),)
        elseif name=="hash64_32.retrieve_batch_masked_kernel"
            @assert n==7;f=lr_s_retrieve_batch_masked_kernel;args=(p(1,UInt64),p(2,UInt32),p(3,UInt64),p(4,UInt32),p(5,Int32),i(6),i(7),)
        elseif name=="hash32_32.getEntryIndicator"
            @assert n==3;f=lr_s_getEntryIndicator;args=(p(1,Int32),p(2,UInt32),i(3),)
        elseif name=="hash32_32.gatherKeyValues"
            @assert n==6;f=lr_s_gatherKeyValues;args=(p(1,Int32),p(2,UInt32),p(3,UInt32),p(4,UInt32),p(5,UInt32),i(6),)
        elseif name=="hash32_32.insert_batch_no_update_kernel"
            @assert n==6;f=lr_s_insert_batch_no_update_kernel;args=(p(1,UInt32),p(2,UInt32),p(3,UInt32),p(4,UInt32),i(5),i(6),)
        elseif name=="hash32_32.insert_batch_no_update_masked_kernel"
            @assert n==7;f=lr_s_insert_batch_no_update_masked_kernel;args=(p(1,UInt32),p(2,UInt32),p(3,UInt32),p(4,UInt32),p(5,Int32),i(6),i(7),)
        elseif name=="hash32_32.retrieve_batch_kernel"
            @assert n==6;f=lr_s_retrieve_batch_kernel;args=(p(1,UInt32),p(2,UInt32),p(3,UInt32),p(4,UInt32),i(5),i(6),)
        elseif name=="hash32_32.retrieve_batch_masked_kernel"
            @assert n==7;f=lr_s_retrieve_batch_masked_kernel;args=(p(1,UInt32),p(2,UInt32),p(3,UInt32),p(4,UInt32),p(5,Int32),i(6),i(7),)
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
@assert ccall(LR_RUN,Cint,(Cstring,Cstring,Ptr{Cvoid},Cint),ARGS[2],ARGS[3]*".native",C_NULL,parse(Cint,ARGS[4]))==0
@assert ccall(LR_RUN,Cint,(Cstring,Cstring,Ptr{Cvoid},Cint),ARGS[2],ARGS[3]*".julia",LR_CB,parse(Cint,ARGS[4]))==0
CUDA.device_synchronize()
a=read(ARGS[3]*".native");b=read(ARGS[3]*".julia");@assert a==b
println("UNIT_SOURCE_FULL_BALANCE_FFI_PASS rawbytes=",length(a)," dispatch=",LR_COUNTS," nativekernel_fallback=false fullresyn2pending=true")
