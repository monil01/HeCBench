# Actual source-recursion unit against independent reachable/dead-reference oracle.
# This probes real CUDA.jl compilation; it cannot certify full resyn2.
using CUDA
include("c_memory.jl");include("generated_cut_helpers.jl")
function probe_mffc!(a,b,refs,cuts,tableSize,tableId,tableNum,out)
    if threadIdx().x==1
        cut=LRPtr{LRCut}(pointer(cuts))[0];cut.nLeaves=UInt32(4)
        for j in Int32(0):Int32(3);cut.leaves[j]=j+Int32(1);end
        out[1]=lr_CalcMFFC(nothing,Int32(13),LRPtr{LRCut}(pointer(cuts)),LRPtr{Int32}(pointer(a)),LRPtr{Int32}(pointer(b)),LRPtr{Int32}(pointer(tableSize)),LRPtr{Int32}(pointer(tableId)),LRPtr{Int32}(pointer(tableNum)),LRPtr{Int32}(pointer(refs)),Int32(13))
    end
    return
end
a=Int32[0,-1,-1,-1,-1,1,3,5,1,1,1,9,7,12];b=Int32[0,-1,-1,-1,-1,2,4,6,2,3,3,10,11,4]
refs=zeros(Int32,14)
for j in 6:14;refs[a[j]+1]+=1;refs[b[j]+1]+=1;end
# Independent destructive reference-count traversal, stopping at the four cut leaves.
r=copy(refs);removed=Int32[]
function visit(node)
    push!(removed,node)
    for c in (a[node+1],b[node+1])
        if c<=4;continue;end
        r[c+1]-=1
        if r[c+1]==0;visit(c);end
    end
end
visit(Int32(13));@assert length(removed)==8
CUDA.device!(0);da=CuArray(a);db=CuArray(b);dr=CuArray(refs);cuts=CUDA.zeros(UInt8,28);size=CUDA.zeros(Int32,1);ids=CUDA.zeros(Int32,8);nums=CUDA.zeros(Int32,8);out=CUDA.zeros(Int32,1)
CUDA.@cuda threads=1 blocks=1 probe_mffc!(da,db,dr,cuts,size,ids,nums,out)
CUDA.device_synchronize();got=Array(out)
@assert got==Int32[8]
@assert Array(size)==Int32[7]
println("UNIT_SOURCE_RECURSIVE_MFFC_PASS saved=",got," independently_removed=",removed," table_ids=",Array(ids)," table_counts=",Array(nums))
