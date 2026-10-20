# Qualified hybrid: authored Julia explicit kernels, original C++ CPU controller
# and native Thrust host library operations/predicates. No native kernel fallback.
using CUDA,Libdl,SHA
include("c_memory.jl")
include("c_library.jl")
include("generated_cut_helpers.jl")
include("strash_helpers.jl")
include("generated_strash.jl")
include("balance_helpers.jl")
include("generated_balance.jl")
include("factor_helpers.jl")
include("generated_factor.jl")
include("mffc_helpers.jl")
include("generated_mffc.jl")
include("manager_kernels.jl")

CUDA.device!(0);CUDA.context()
const LR_LIBRARY_PATH=abspath(ARGS[1]);const LR_LIB=Libdl.dlopen(LR_LIBRARY_PATH)
const LR_RUN=Libdl.dlsym(LR_LIB,:lr_full_study);const LR_SYMBOL=Libdl.dlsym(LR_LIB,:lr_symbol)
const LR_GLOBALS=(P=lr_ptr(Int32,UInt64(UInt(ccall(LR_SYMBOL,Ptr{Cvoid},(Cint,),0)))),N=lr_ptr(Int32,UInt64(UInt(ccall(LR_SYMBOL,Ptr{Cvoid},(Cint,),1)))))
@assert UInt(getfield(LR_GLOBALS.P,:base))!=0&&UInt(getfield(LR_GLOBALS.N,:base))!=0
const LR_COUNTS=Dict{String,Int}()
Base.@noinline function lr_launch_device(f,args,threads,grid)
    CUDA.@cuda threads=threads blocks=grid stream=CUDA.legacy_stream() maxregs=128 f(LR_GLOBALS,args...)
    return nothing
end
function lr_callback(d::Ptr{Cvoid})::Cint
 name="packet_unparsed"
 try
    name=unsafe_string(unsafe_load(Ptr{Cstring}(d)));base=split(split(name,'.')[end],"::")[end];words=Ptr{UInt64}(d)
    grid=ntuple(i->Int(unsafe_load(words,1+i)),3);threads=ntuple(i->Int(unsafe_load(words,4+i)),3);n=Int(unsafe_load(words,8));@assert n<=32
    v=ntuple(i->unsafe_load(words,40+i),n);i(k)=lr_cast(Int32,v[k]);p(k,T)=lr_ptr(T,v[k])
        if base=="BuildHashTable"
            @assert n==6;f=lr_BuildHashTable;args=(p(1,LRTableNode),lr_cast(Int32,v[2]),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),)
        elseif base=="Convert"
            @assert n==3;f=lr_Convert;args=(p(1,Int32),p(2,Int32),lr_cast(Int32,v[3]),)
        elseif base=="CutEnumerate"
            @assert n==8;f=lr_CutEnumerate;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,LRCut),lr_cast(Int32,v[7]),lr_cast(Int32,v[8]),)
        elseif base=="DetachAndAttach"
            @assert n==4;f=lr_DetachAndAttach;args=(lr_cast(Int32,v[1]),p(2,Int32),p(3,Int32),p(4,Int32),)
        elseif base=="EvaluateNode"
            @assert n==13;f=lr_EvaluateNode;args=(lr_cast(Int32,v[1]),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,LRCut),p(9,LRCut),p(10,Int32),p(11,LRLibrary),p(12,LRTableNode),lr_cast(Int32,v[13]),)
        elseif base=="Inputs"
            @assert n==3;f=lr_Inputs;args=(p(1,Int32),p(2,LRCut),lr_cast(Int32,v[3]),)
        elseif base=="ReplaceSubgr"
            @assert n==12;f=lr_ReplaceSubgr;args=(lr_cast(Int32,v[1]),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,LRCut),p(8,LRLibrary),p(9,LRTableNode),p(10,LRTableNode),p(11,Int32),p(12,Int32),)
        elseif base=="Revert"
            @assert n==3;f=lr_Revert;args=(p(1,Int32),p(2,Int32),lr_cast(Int32,v[3]),)
        elseif base=="addBackLocalArrays"
            @assert n==8;f=lr_addBackLocalArrays;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt32),p(5,Int32),p(6,UInt32),lr_cast(Int32,v[7]),lr_cast(Int32,v[8]),)
        elseif base=="assignOld2NewConsecutiveIds"
            @assert n==4;f=lr_assignOld2NewConsecutiveIds;args=(p(1,UInt32),p(2,Int32),lr_cast(Int32,v[3]),lr_cast(Int32,v[4]),)
        elseif base=="buildHashTable"
            @assert n==7;f=lr_buildHashTable;args=(p(1,Int32),p(2,Int32),p(3,UInt64),p(4,UInt32),lr_cast(Int32,v[5]),lr_cast(Int32,v[6]),lr_cast(Int32,v[7]),)
        elseif base=="checkCoverTravEnd"
            @assert n==6;f=lr_checkCoverTravEnd;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),lr_cast(Int32,v[5]),lr_cast(Int32,v[6]),)
        elseif base=="checkInsertion"
            @assert n==4;f=lr_checkInsertion;args=(p(1,Int32),p(2,Int32),p(3,Int32),lr_cast(Int32,v[4]),)
        elseif base=="checkSingleVarSubg"
            @assert n==9;f=lr_checkSingleVarSubg;args=(p(1,UInt64),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),lr_cast(Int32,v[8]),lr_cast(Int32,v[9]),)
        elseif base=="coverFindingToTable"
            @assert n==13;f=lr_coverFindingToTable;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,Int32),p(9,Int32),p(10,Int32),lr_cast(Int32,v[11]),lr_cast(Int32,v[12]),lr_cast(Int32,v[13]),)
        elseif base=="duplicateHashTableWithoutMFFCs"
            @assert n==8;f=lr_duplicateHashTableWithoutMFFCs;args=(p(1,Int32),p(2,Int32),p(3,UInt64),p(4,UInt32),lr_cast(Int32,v[5]),p(6,UInt64),p(7,UInt32),lr_cast(Int32,v[8]),)
        elseif base=="evalFactoredForm"
            @assert n==14;f=lr_evalFactoredForm;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,UInt64),p(8,UInt32),lr_cast(Int32,v[9]),p(10,UInt64),p(11,Int32),p(12,Int32),p(13,Int32),lr_cast(Int32,v[14]),)
        elseif base=="factorFromTruth"
            @assert n==11;f=lr_factorFromTruth;args=(p(1,Int32),p(2,Int32),p(3,UInt64),p(4,Int32),p(5,Int32),p(6,Int32),p(7,UInt32),p(8,UInt32),p(9,Int32),p(10,UInt32),lr_cast(Int32,v[11]),)
        elseif base=="findLevelNodeRanges"
            @assert n==3;f=lr_findLevelNodeRanges;args=(p(1,Int32),p(2,Int32),lr_cast(Int32,v[3]),)
        elseif base=="gatherByScannedMask"
            @assert n==4;f=lr_gatherByScannedMask;args=(p(1,Int32),p(2,Int32),lr_cast(Int32,v[3]),lr_cast(Int32,v[4]),)
        elseif base=="gatherTableToConsecutive"
            @assert n==6;f=lr_gatherTableToConsecutive;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),lr_cast(Int32,v[6]),)
        elseif base=="gatherWithFilter"
            @assert n==10;f=lr_gatherWithFilter;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,Int32),lr_cast(Int32,v[9]),lr_cast(Int32,v[10]),)
        elseif base=="genReadyMask"
            @assert n==9;f=lr_genReadyMask;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),lr_cast(Int32,v[8]),lr_cast(Int32,v[9]),)
        elseif base=="getCoverToNodeIdMapping"
            @assert n==4;f=lr_getCoverToNodeIdMapping;args=(p(1,Int32),p(2,Int32),p(3,Int32),lr_cast(Int32,v[4]),)
        elseif base=="getCutTruthRanges"
            @assert n==5;f=lr_getCutTruthRanges;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),lr_cast(Int32,v[5]),)
        elseif base=="getCutTruthTableConsecutive"
            @assert n==13;f=lr_getCutTruthTableConsecutive;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,UInt32),p(8,Int32),p(9,UInt32),lr_cast(Int32,v[10]),lr_cast(Int32,v[11]),lr_cast(Int32,v[12]),p(13,Int32),)
        elseif base=="getElemTruthTable"
            @assert n==2;f=lr_getElemTruthTable;args=(p(1,UInt32),lr_cast(Int32,v[2]),)
        elseif base=="insertLevelNodes"
            @assert n==10;f=lr_insertLevelNodes;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,UInt64),p(6,UInt32),lr_cast(Int32,v[7]),p(8,UInt64),lr_cast(Int32,v[9]),lr_cast(Int32,v[10]),)
        elseif base=="insertSubgIter"
            @assert n==13;f=lr_insertSubgIter;args=(lr_cast(Int32,v[1]),p(2,Int32),p(3,Int32),p(4,Int32),p(5,UInt64),p(6,UInt32),lr_cast(Int32,v[7]),p(8,UInt64),p(9,Int32),p(10,Int32),p(11,Int32),lr_cast(Int32,v[12]),lr_cast(Int32,v[13]),)
        elseif base=="markDanglingNodesIter"
            @assert n==6;f=lr_markDanglingNodesIter;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),lr_cast(Int32,v[5]),lr_cast(Int32,v[6]),)
        elseif base=="markIsCoverOutput"
            @assert n==3;f=lr_markIsCoverOutput;args=(p(1,Int32),p(2,Int32),lr_cast(Int32,v[3]),)
        elseif base=="markReadyNodes"
            @assert n==6;f=lr_markReadyNodes;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),lr_cast(Int32,v[6]),)
        elseif base=="parseOutputRes"
            @assert n==9;f=lr_parseOutputRes;args=(p(1,UInt64),p(2,UInt32),p(3,UInt32),lr_cast(Int32,v[4]),p(5,Int32),p(6,Int32),p(7,Int32),lr_cast(Int32,v[8]),lr_cast(Int32,v[9]),)
        elseif base=="poReId"
            @assert n==4;f=lr_poReId;args=(p(1,Int32),p(2,Int32),p(3,Int32),lr_cast(Int32,v[4]),)
        elseif base=="poUpdateId"
            @assert n==4;f=lr_poUpdateId;args=(p(1,Int32),p(2,Int32),p(3,Int32),lr_cast(Int32,v[4]),)
        elseif base=="prepareDataToInsert"
            @assert n==18;f=lr_prepareDataToInsert;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt64),p(5,UInt32),p(6,Int32),p(7,UInt32),p(8,UInt64),p(9,UInt32),lr_cast(Int32,v[10]),p(11,UInt32),p(12,UInt32),lr_cast(Int32,v[13]),lr_cast(Int32,v[14]),lr_cast(Int32,v[15]),lr_cast(Int32,v[16]),lr_cast(Int32,v[17]),lr_cast(Int32,v[18]),)
        elseif base=="prepareReconstructArrays"
            @assert n==17;f=lr_prepareReconstructArrays;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,UInt32),p(9,UInt32),lr_cast(Int32,v[10]),p(11,Int32),p(12,Int32),p(13,Int32),lr_cast(Int32,v[14]),lr_cast(Int32,v[15]),lr_cast(Int32,v[16]),lr_cast(Int32,v[17]),)
        elseif base=="printStatsKernel"
            @assert n==3;f=lr_printStatsKernel;args=(p(1,Int32),p(2,Int32),p(3,Int32),)
        elseif base=="processPO"
            @assert n==10;f=lr_processPO;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt32),p(5,UInt32),lr_cast(Int32,v[6]),p(7,Int32),p(8,Int32),lr_cast(Int32,v[9]),lr_cast(Int32,v[10]),)
        elseif base=="processRwmanFanins"
            @assert n==5;f=lr_processRwmanFanins;args=(p(1,Int32),p(2,Int32),p(3,Int32),lr_cast(Int32,v[4]),lr_cast(Int32,v[5]),)
        elseif base=="processRwmanOuts"
            @assert n==3;f=lr_processRwmanOuts;args=(p(1,Int32),p(2,Int32),lr_cast(Int32,v[3]),)
        elseif base=="recordMFFC"
            @assert n==11;f=lr_recordMFFC;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,Int32),lr_cast(Int32,v[9]),lr_cast(Int32,v[10]),lr_cast(Int32,v[11]),)
        elseif base=="recordReconstructedCovers"
            @assert n==7;f=lr_recordReconstructedCovers;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),lr_cast(Int32,v[6]),lr_cast(Int32,v[7]),)
        elseif base=="setStatus"
            @assert n==6;f=lr_setStatus;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),lr_cast(Int32,v[5]),lr_cast(Int32,v[6]),)
        elseif base=="sharedNodeDrivenPermute"
            @assert n==8;f=lr_sharedNodeDrivenPermute;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,UInt64),p(5,UInt32),lr_cast(Int32,v[6]),lr_cast(Int32,v[7]),lr_cast(Int32,v[8]),)
        elseif base=="unbindKeysReId"
            @assert n==6;f=lr_unbindKeysReId;args=(p(1,UInt64),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),lr_cast(Int32,v[6]),)
        elseif base=="unbindKeysUpdateOldRoots"
            @assert n==8;f=lr_unbindKeysUpdateOldRoots;args=(p(1,Int32),p(2,UInt64),p(3,UInt32),p(4,Int32),p(5,Int32),lr_cast(Int32,v[6]),lr_cast(Int32,v[7]),lr_cast(Int32,v[8]),)
        elseif base=="updateDeviceStats"
            @assert n==4;f=lr_updateDeviceStats;args=(lr_cast(Int32,v[1]),lr_cast(Int32,v[2]),p(3,Int32),p(4,Int32),)
        elseif base=="updateInsertedIdsIter"
            @assert n==13;f=lr_updateInsertedIdsIter;args=(lr_cast(Int32,v[1]),p(2,Int32),p(3,Int32),p(4,UInt64),p(5,UInt32),lr_cast(Int32,v[6]),p(7,UInt64),p(8,Int32),p(9,Int32),p(10,Int32),p(11,Int32),p(12,Int32),lr_cast(Int32,v[13]),)
        elseif base=="updateLevelNodesNewIds"
            @assert n==7;f=lr_updateLevelNodesNewIds;args=(p(1,Int32),p(2,Int32),p(3,UInt64),p(4,UInt64),p(5,UInt32),lr_cast(Int32,v[6]),lr_cast(Int32,v[7]),)
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
            error("No authored Julia implementation for intercepted explicit source kernel: ",name)
        end
    Base.invokelatest(lr_launch_device,f,args,threads,grid)
    LR_COUNTS[name]=get(LR_COUNTS,name,0)+1
    return Cint(0)
 catch err
    println(stderr,"JULIA_SOURCE_KERNEL_REJECTED name=",name," completed_dispatches=",LR_COUNTS)
    showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);return Cint(1)
 end
end
precompile(lr_callback,(Ptr{Cvoid},));const LR_CALLBACK=@cfunction(lr_callback,Cint,(Ptr{Cvoid},))
println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(LR_LIBRARY_PATH)))," explicit_kernel_native_fallback=false; original_native_host_Thrust_library_interoperation=true")
status=ccall(LR_RUN,Cint,(Cstring,Ptr{Cvoid}),abspath(ARGS[2]),LR_CALLBACK)
CUDA.device_synchronize();println("JULIA_EXPLICIT_KERNEL_DISPATCH_COUNTS ",LR_COUNTS)
status==0||error("Full source pipeline failed its declared checks/status ",status)
