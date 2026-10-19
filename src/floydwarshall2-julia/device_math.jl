# Source statement transcription: zero-based adapters retain original indexing.
const ws=Int32(32)
const tile=Int32(64)
const ThreadsPerBlock=Int32(1024)
const INT_MAX=typemax(Int32)
struct FZero{A}
    a::A
    offset::Int32
end
FZero(a)=FZero(a,Int32(0))
@inline Base.getindex(a::FZero,i::Integer)=(@inbounds a.a[i+a.offset+Int32(1)])
@inline Base.setindex!(a::FZero,v,i::Integer)=(@inbounds a.a[i+a.offset+Int32(1)]=v)
@inline f_shuffle(v,k)=CUDA.shfl_sync(UInt32(0xffffffff),v,k+Int32(1))


@inline function f_init1!(nodes,AdjMat,upper,tid,block)
idx = tid + block * ThreadsPerBlock
i = idx ÷ upper
if i < upper
j = idx % upper
AdjMat[idx] = ((i == j) && (i < nodes)) ? Int32(0) : (INT_MAX ÷ Int32(2))
end
nothing
end

@inline function f_init2!(nodes,csr_idx,csr_nei,csr_wt,AdjMat,upper,tid,block)
i = tid + block * ThreadsPerBlock
if i < nodes
for j in csr_idx[i]:(csr_idx[i+Int32(1)]-Int32(1))
nei = csr_nei[j]
AdjMat[i * upper + nei] = csr_wt[j]
end
end
nothing
end

@inline function f_FW0_64!(AdjMat,upper,krows,kcols,tid,block)
scratch=CUDA.CuStaticSharedArray(Int32,8192)
temp=FZero(scratch,Int32(0))
krow=FZero(scratch,Int32(4096))
warp_a = tid ÷ ws
warp_b = warp_a + ws
lane_a = tid % ws
lane_b = lane_a + ws
idx0_aa = warp_a * upper + lane_a
idx0_ab = warp_a * upper + lane_b
idx0_ba = warp_b * upper + lane_a
idx0_bb = warp_b * upper + lane_b
idx1_aa = lane_a * tile + warp_a
idx1_ab = lane_b * tile + warp_a
idx1_ba = lane_a * tile + warp_b
idx1_bb = lane_b * tile + warp_b
idx2_a = lane_a
idx2_b = lane_b
ij_aa = AdjMat[idx0_aa]
ij_ab = AdjMat[idx0_ab]
ij_ba = AdjMat[idx0_ba]
ij_bb = AdjMat[idx0_bb]
Base.Cartesian.@nexprs 64 ku -> begin
k=Int32(ku-1)
if warp_a == k
krow[idx2_a] = ij_aa
end
if warp_a == k
krow[idx2_b] = ij_ab
end
if warp_b == k
krow[idx2_a] = ij_ba
end
if warp_b == k
krow[idx2_b] = ij_bb
end
CUDA.sync_threads()
ik_a=Int32(0)
ik_b=Int32(0)
if k < ws
ik_a = f_shuffle(ij_aa, k)
ik_b = f_shuffle(ij_ba, k)
else
ik_a = f_shuffle(ij_ab, k - ws)
ik_b = f_shuffle(ij_bb, k - ws)
end
kr_a = krow[idx2_a]
kr_b = krow[idx2_b]
ij_aa = min(ij_aa, ik_a + kr_a)
ij_ab = min(ij_ab, ik_a + kr_b)
ij_ba = min(ij_ba, ik_b + kr_a)
ij_bb = min(ij_bb, ik_b + kr_b)
if warp_a == k
krows[idx0_aa] = ij_aa
end
if warp_a == k
krows[idx0_ab] = ij_ab
end
if warp_b == k
krows[idx0_ba] = ij_ba
end
if warp_b == k
krows[idx0_bb] = ij_bb
end
if lane_a == k
temp[idx1_aa] = ij_aa
end
if lane_a == k
temp[idx1_ba] = ij_ba
end
if lane_b == k
temp[idx1_ab] = ij_ab
end
if lane_b == k
temp[idx1_bb] = ij_bb
end
idx2_a += tile
idx2_b += tile
end
CUDA.sync_threads()
kcols[idx0_aa] = temp[warp_a * tile + lane_a]
kcols[idx0_ab] = temp[warp_a * tile + lane_b]
kcols[idx0_ba] = temp[warp_b * tile + lane_a]
kcols[idx0_bb] = temp[warp_b * tile + lane_b]
AdjMat[idx0_aa] = ij_aa
AdjMat[idx0_ab] = ij_ab
AdjMat[idx0_ba] = ij_ba
AdjMat[idx0_bb] = ij_bb
nothing
end

@inline function f_FWrowcol_64!(AdjMat,upper,krows,kcols,x,subm1,tid,block)
scratch=CUDA.CuStaticSharedArray(Int32,8192)
temp=FZero(scratch,Int32(0))
krow=FZero(scratch,Int32(4096))
warp_a = tid ÷ ws
warp_b = warp_a + ws
lane_a = tid % ws
lane_b = lane_a + ws
y = block
if y < subm1
if y >= x
y += Int32(1)
end
i_a = warp_a + x * tile
i_b = warp_b + x * tile
j_a = lane_a + y * tile
j_b = lane_b + y * tile
idx0_aa = i_a * upper + j_a
idx0_ab = i_a * upper + j_b
idx0_ba = i_b * upper + j_a
idx0_bb = i_b * upper + j_b
idx1_a = warp_a
idx1_b = warp_b
idx2_a = lane_a
idx2_b = lane_b
temp[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile]
temp[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile]
temp[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile]
temp[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile]
CUDA.sync_threads()
ij_aa = AdjMat[idx0_aa]
ij_ab = AdjMat[idx0_ab]
ij_ba = AdjMat[idx0_ba]
ij_bb = AdjMat[idx0_bb]
orig_aa = ij_aa
orig_ab = ij_ab
orig_ba = ij_ba
orig_bb = ij_bb
Base.Cartesian.@nexprs 64 ku -> begin
k=Int32(ku-1)
if warp_a == k
krow[idx2_a] = ij_aa
end
if warp_a == k
krow[idx2_b] = ij_ab
end
if warp_b == k
krow[idx2_a] = ij_ba
end
if warp_b == k
krow[idx2_b] = ij_bb
end
CUDA.sync_threads()
ik_a = temp[idx1_a]
ik_b = temp[idx1_b]
kr_a = krow[idx2_a]
kr_b = krow[idx2_b]
ij_aa = min(ij_aa, ik_a + kr_a)
ij_ab = min(ij_ab, ik_a + kr_b)
ij_ba = min(ij_ba, ik_b + kr_a)
ij_bb = min(ij_bb, ik_b + kr_b)
if warp_a == k
krows[idx0_aa] = ij_aa
end
if warp_a == k
krows[idx0_ab] = ij_ab
end
if warp_b == k
krows[idx0_ba] = ij_ba
end
if warp_b == k
krows[idx0_bb] = ij_bb
end
idx1_a += tile
idx1_b += tile
idx2_a += tile
idx2_b += tile
end
if ij_aa != orig_aa
AdjMat[idx0_aa] = ij_aa
end
if ij_ab != orig_ab
AdjMat[idx0_ab] = ij_ab
end
if ij_ba != orig_ba
AdjMat[idx0_ba] = ij_ba
end
if ij_bb != orig_bb
AdjMat[idx0_bb] = ij_bb
end
else
y -= subm1
if y >= x
y += Int32(1)
end
i_a = warp_a + y * tile
i_b = warp_b + y * tile
j_a = lane_a + x * tile
j_b = lane_b + x * tile
idx0_aa = i_a * upper + j_a
idx0_ab = i_a * upper + j_b
idx0_ba = i_b * upper + j_a
idx0_bb = i_b * upper + j_b
idx1_aa = lane_a * tile + warp_a
idx1_ab = lane_b * tile + warp_a
idx1_ba = lane_a * tile + warp_b
idx1_bb = lane_b * tile + warp_b
idx2_a = (x * tile) * upper + j_a
idx2_b = (x * tile) * upper + j_b
ij_aa = AdjMat[idx0_aa]
ij_ab = AdjMat[idx0_ab]
ij_ba = AdjMat[idx0_ba]
ij_bb = AdjMat[idx0_bb]
orig_aa = ij_aa
orig_ab = ij_ab
orig_ba = ij_ba
orig_bb = ij_bb
Base.Cartesian.@nexprs 64 ku -> begin
k=Int32(ku-1)
ik_a=Int32(0)
ik_b=Int32(0)
if k < ws
ik_a = f_shuffle(ij_aa, k)
ik_b = f_shuffle(ij_ba, k)
end
if k >= ws
ik_a = f_shuffle(ij_ab, k - ws)
ik_b = f_shuffle(ij_bb, k - ws)
end
kr_a = krows[idx2_a]
kr_b = krows[idx2_b]
ij_aa = min(ij_aa, ik_a + kr_a)
ij_ab = min(ij_ab, ik_a + kr_b)
ij_ba = min(ij_ba, ik_b + kr_a)
ij_bb = min(ij_bb, ik_b + kr_b)
if lane_a == k
temp[idx1_aa] = ij_aa
end
if lane_a == k
temp[idx1_ba] = ij_ba
end
if lane_b == k
temp[idx1_ab] = ij_ab
end
if lane_b == k
temp[idx1_bb] = ij_bb
end
idx2_a += upper
idx2_b += upper
end
CUDA.sync_threads()
kcols[idx0_aa] = temp[warp_a * tile + lane_a]
kcols[idx0_ab] = temp[warp_a * tile + lane_b]
kcols[idx0_ba] = temp[warp_b * tile + lane_a]
kcols[idx0_bb] = temp[warp_b * tile + lane_b]
if ij_aa != orig_aa
AdjMat[idx0_aa] = ij_aa
end
if ij_ab != orig_ab
AdjMat[idx0_ab] = ij_ab
end
if ij_ba != orig_ba
AdjMat[idx0_ba] = ij_ba
end
if ij_bb != orig_bb
AdjMat[idx0_bb] = ij_bb
end
end
nothing
end

@inline function f_FWrem_64!(AdjMat,upper,krows,kcols,x,subm1,tid,block)
y = block ÷ subm1
z = block % subm1
if y >= x
y += Int32(1)
end
if z >= x
z += Int32(1)
end
warp_a = tid ÷ ws
warp_b = warp_a + ws
lane_a = tid % ws
lane_b = lane_a + ws
i_a = warp_a + y * tile
i_b = warp_b + y * tile
j_a = lane_a + z * tile
j_b = lane_b + z * tile
idx0_aa = i_a * upper + j_a
idx0_ab = i_a * upper + j_b
idx0_ba = i_b * upper + j_a
idx0_bb = i_b * upper + j_b
scratch=CUDA.CuStaticSharedArray(Int32,8192)
s_kj=FZero(scratch,Int32(0))
s_ik=FZero(scratch,Int32(4096))
s_kj[warp_a * tile + lane_a] = krows[(x * tile + warp_a) * upper + j_a]
s_kj[warp_a * tile + lane_b] = krows[(x * tile + warp_a) * upper + j_b]
s_kj[warp_b * tile + lane_a] = krows[(x * tile + warp_b) * upper + j_a]
s_kj[warp_b * tile + lane_b] = krows[(x * tile + warp_b) * upper + j_b]
s_ik[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile]
s_ik[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile]
s_ik[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile]
s_ik[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile]
ij_aa = AdjMat[idx0_aa]
ij_ab = AdjMat[idx0_ab]
ij_ba = AdjMat[idx0_ba]
ij_bb = AdjMat[idx0_bb]
orig_aa = ij_aa
orig_ab = ij_ab
orig_ba = ij_ba
orig_bb = ij_bb
CUDA.sync_threads()
idx1_a = warp_a
idx1_b = warp_b
idx2_a = lane_a
idx2_b = lane_b
Base.Cartesian.@nexprs 64 ku -> begin
k=Int32(ku-1)
sk_a = s_kj[idx2_a]
sk_b = s_kj[idx2_b]
ij_aa = min(ij_aa, s_ik[idx1_a] + sk_a)
ij_ab = min(ij_ab, s_ik[idx1_a] + sk_b)
ij_ba = min(ij_ba, s_ik[idx1_b] + sk_a)
ij_bb = min(ij_bb, s_ik[idx1_b] + sk_b)
idx1_a += tile
idx1_b += tile
idx2_a += tile
idx2_b += tile
end
CUDA.sync_threads()
if (y == z) && (y == x + Int32(1)) && (x != subm1)
idx1_aa = lane_a * tile + warp_a
idx1_ab = lane_b * tile + warp_a
idx1_ba = lane_a * tile + warp_b
idx1_bb = lane_b * tile + warp_b
idx2_a = lane_a
idx2_b = lane_b
Base.Cartesian.@nexprs 64 ku -> begin
k=Int32(ku-1)
if warp_a == k
s_kj[idx2_a] = ij_aa
end
if warp_a == k
s_kj[idx2_b] = ij_ab
end
if warp_b == k
s_kj[idx2_a] = ij_ba
end
if warp_b == k
s_kj[idx2_b] = ij_bb
end
ik_a=Int32(0)
ik_b=Int32(0)
if k < ws
ik_a = f_shuffle(ij_aa, k)
ik_b = f_shuffle(ij_ba, k)
else
ik_a = f_shuffle(ij_ab, k - ws)
ik_b = f_shuffle(ij_bb, k - ws)
end
CUDA.sync_threads()
sk_a = s_kj[idx2_a]
sk_b = s_kj[idx2_b]
ij_aa = min(ij_aa, ik_a + sk_a)
ij_ab = min(ij_ab, ik_a + sk_b)
ij_ba = min(ij_ba, ik_b + sk_a)
ij_bb = min(ij_bb, ik_b + sk_b)
if warp_a == k
krows[idx0_aa] = ij_aa
end
if warp_a == k
krows[idx0_ab] = ij_ab
end
if warp_b == k
krows[idx0_ba] = ij_ba
end
if warp_b == k
krows[idx0_bb] = ij_bb
end
if lane_a == k
s_ik[idx1_aa] = ij_aa
end
if lane_a == k
s_ik[idx1_ba] = ij_ba
end
if lane_b == k
s_ik[idx1_ab] = ij_ab
end
if lane_b == k
s_ik[idx1_bb] = ij_bb
end
idx2_a += tile
idx2_b += tile
end
CUDA.sync_threads()
kcols[idx0_aa] = s_ik[warp_a * tile + lane_a]
kcols[idx0_ab] = s_ik[warp_a * tile + lane_b]
kcols[idx0_ba] = s_ik[warp_b * tile + lane_a]
kcols[idx0_bb] = s_ik[warp_b * tile + lane_b]
end
if ij_aa != orig_aa
AdjMat[idx0_aa] = ij_aa
end
if ij_ab != orig_ab
AdjMat[idx0_ab] = ij_ab
end
if ij_ba != orig_ba
AdjMat[idx0_ba] = ij_ba
end
if ij_bb != orig_bb
AdjMat[idx0_bb] = ij_bb
end
nothing
end
