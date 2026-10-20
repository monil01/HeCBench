# Source Rewrite GPU core + original CPU reconstruction operator unit only.
# All intercepted kernels execute authored Julia bodies; no native fallback.
using CUDA,Libdl
include("c_memory.jl");include("c_library.jl");include("generated_cut_helpers.jl")
CUDA.device!(0);CUDA.context()
const LRLIB=Libdl.dlopen(abspath(ARGS[1]));const LR_RUN=Libdl.dlsym(LRLIB,:lr_rewrite_unit);const LR_SYMBOL=Libdl.dlsym(LRLIB,:lr_symbol)
@assert ccall(Libdl.dlsym(LRLIB,:lr_descriptor_size),Csize_t,())==576
const LR_COUNTS=Dict{String,Int}()
function lr_callback(d::Ptr{Cvoid})::Cint
    try
        name=unsafe_string(unsafe_load(Ptr{Cstring}(d)))
        words=Ptr{UInt64}(d);grid=ntuple(i->Int(unsafe_load(words,1+i)),3);threads=ntuple(i->Int(unsafe_load(words,4+i)),3);n=Int(unsafe_load(words,8));@assert n<=32
        v=ntuple(i->unsafe_load(words,40+i),n)
        i(k)=lr_cast(Int32,v[k]);p(k,T)=lr_ptr(T,v[k])
        # Registered native managed symbols are actual storage, not copied guesses.
        globals=(P=lr_ptr(Int32,UInt64(UInt(ccall(LR_SYMBOL,Ptr{Cvoid},(Cint,),0)))),N=lr_ptr(Int32,UInt64(UInt(ccall(LR_SYMBOL,Ptr{Cvoid},(Cint,),1)))))
        f=nothing;args=()
        if name=="Convert" || name=="Revert"
            @assert n==3;f=name=="Convert" ? lr_Convert : lr_Revert;args=(p(1,Int32),p(2,Int32),i(3))
        elseif name=="Inputs"
            @assert n==3;f=lr_Inputs;args=(p(1,Int32),p(2,LRCut),i(3))
        elseif name=="CutEnumerate"
            @assert n==8;f=lr_CutEnumerate;args=(p(1,Int32),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,LRCut),i(7),i(8))
        elseif name=="BuildHashTable"
            @assert n==6;f=lr_BuildHashTable;args=(p(1,LRTableNode),i(2),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32))
        elseif name=="EvaluateNode"
            @assert n==13;f=lr_EvaluateNode;args=(i(1),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,Int32),p(8,LRCut),p(9,LRCut),p(10,Int32),p(11,LRLibrary),p(12,LRTableNode),i(13))
        elseif name=="ReplaceSubgr"
            @assert n==12;f=lr_ReplaceSubgr;args=(i(1),p(2,Int32),p(3,Int32),p(4,Int32),p(5,Int32),p(6,Int32),p(7,LRCut),p(8,LRLibrary),p(9,LRTableNode),p(10,LRTableNode),p(11,Int32),p(12,Int32))
        else
            error("Source kernel has no authored Julia dispatcher: ",name)
        end
        @assert grid[2:3]==(1,1)&&threads[2:3]==(1,1)
        # Same legacy stream as the original CUDA Runtime host controller.
        println("CALLBACK_ABI_DIAGNOSTIC ",(name=name,grid=grid,threads=threads,nargs=n,wire=v,globals=globals));flush(stdout);return Cint(1)
        LR_COUNTS[name]=get(LR_COUNTS,name,0)+1
        return Cint(0)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);return Cint(1)
    end
end
const LR_CB=@cfunction(lr_callback,Cint,(Ptr{Cvoid},))
println("DIAGNOSTIC no native remeasurement")
CUDA.device_synchronize()
rc=ccall(LR_RUN,Cint,(Cstring,Cstring,Ptr{Cvoid}),ARGS[2],ARGS[3]*".julia",LR_CB);println("EXPECTED_CALLBACK_REJECTION ",rc);exit(rc==1 ? 0 : 1)
CUDA.device_synchronize()
a=read(ARGS[3]*".native");b=read(ARGS[3]*".julia")
if a!=b
    println("REWRITE_CORE_RAW_MISMATCH nativebytes=",length(a)," juliabytes=",length(b));error("sourceRewriteGPUcoreunit mismatchesoriginalCUDA")
end
println("UNIT_SOURCE_REWRITE_CORE_FFI_PASS rawbytes=",length(a)," dispatch=",LR_COUNTS," authorednativekernelfallback=false; fullresyn2pending")
