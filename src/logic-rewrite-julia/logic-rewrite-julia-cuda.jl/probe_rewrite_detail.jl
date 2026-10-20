# Genuine source rewrite pre-evaluation subsystem unit; full resyn2 pending.
using CUDA
include("c_memory.jl");include("c_library.jl");include("generated_cut_helpers_detail.jl")
function run_convert!(a,c,n)
    lr_Convert(nothing,LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(c)),n);return
end
function run_inputs!(refs,cuts,n)
    lr_Inputs(nothing,LRPtr{Int32}(pointer(refs)),LRPtr{LRCut}(pointer(cuts)),n);return
end
function run_enum!(a,b,c0,c1,refs,cuts,delta,n)
    lr_CutEnumerate(nothing,LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(b)),LRPtr{Int32}(pointer(c0)),LRPtr{Int32}(pointer(c1)),LRPtr{Int32}(pointer(refs)),LRPtr{LRCut}(pointer(cuts)),delta,n);return
end
function context(g)
    (P=LRPtr{Int32}(pointer(g)),N=LRPtr{Int32}(pointer(g)+4))
end
function run_hash!(table,n,a,b,c0,c1,g)
    lr_BuildHashTable(context(g),LRPtr{LRTableNode}(pointer(table)),n,LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(b)),LRPtr{Int32}(pointer(c0)),LRPtr{Int32}(pointer(c1)));return
end
function run_evaluate!(n,best,a,b,c0,c1,level,cuts,selected,refs,lib,table,g)
    lr_EvaluateNode(context(g),n,LRPtr{Int32}(pointer(best)),LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(b)),LRPtr{Int32}(pointer(c0)),LRPtr{Int32}(pointer(c1)),LRPtr{Int32}(pointer(level)),LRPtr{LRCut}(pointer(cuts)),LRPtr{LRCut}(pointer(selected)),LRPtr{Int32}(pointer(refs)),LRPtr{LRLibrary}(pointer(lib)),LRPtr{LRTableNode}(pointer(table)),Int32(0));return
end
f=open(ARGS[1]);n=read(f,Int32);pi=read(f,Int32);len=read(f,Int32);counts=read!(f,Vector{Int32}(undef,len));a=read!(f,Vector{Int32}(undef,n+1));b=read!(f,similar(a));levels=read!(f,similar(a));refs=read!(f,similar(a));lib=read(f,1019472);expected=read(f);close(f)
CUDA.device!(0);da=CuArray(a);db=CuArray(b);dr=CuArray(refs);dl=CuArray(levels);dlib=CuArray(lib);c0=CUDA.zeros(Int32,n+1);c1=CUDA.zeros(Int32,n+1);cuts=CUDA.zeros(UInt8,(n+1)*8*28);selected=CUDA.zeros(UInt8,(n+1)*28);best=CUDA.zeros(Int32,n+1)
P=Int32(17);table=CUDA.fill(UInt8(255),(2*n+1+P)*8);g=CuArray(Int32[P,n])
@assert Array(da)==a&&Array(db)==b&&Array(dr)==refs&&Array(dl)==levels&&Array(dlib)==lib
CUDA.@cuda threads=512 blocks=1 always_inline=true run_convert!(da,c0,n+Int32(1))
CUDA.@cuda threads=512 blocks=1 always_inline=true run_convert!(db,c1,n+Int32(1))
CUDA.@cuda threads=512 blocks=1 always_inline=true run_inputs!(dr,cuts,pi)
for k in 2:length(counts)
    CUDA.@cuda threads=512 blocks=1 maxregs=128 always_inline=true run_enum!(da,db,c0,c1,dr,cuts,counts[k-1],counts[k]-counts[k-1])
end
CUDA.@cuda threads=512 blocks=1 maxregs=128 always_inline=true run_hash!(table,n,da,db,c0,c1,g)
CUDA.@cuda threads=512 blocks=1 maxregs=128 run_evaluate!(n,best,da,db,c0,c1,dl,cuts,selected,dr,dlib,table,g)
CUDA.device_synchronize()
got=IOBuffer();for x in (da,db,c0,c1,cuts,selected,best);write(got,Array(x));end;raw=take!(got);write(ARGS[2],raw)
if raw!=expected
    ix=findall(raw.!=expected);println("REWRITE_PRE_UNIT_DIFFS ",length(ix)," first=",ix[1:min(end,20)])
    for j in ix[1:min(end,20)];println("byte ",j," got=",raw[j]," native=",expected[j]);end
    error("full source rewrite pre-evaluation subsystem mismatch")
end
println("UNIT_SOURCE_REWRITE_PRE_EVALUATION_PASS ",length(raw)," rawbytes, sourceNPNlibrary1019472bytes")
