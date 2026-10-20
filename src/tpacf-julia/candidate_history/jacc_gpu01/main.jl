using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include(joinpath(@__DIR__,"..","study_device.jl"))
# Lazy source AoS double3 shared tile; no materialized scratch/global array.
struct TileVector{A,B,C} <: AbstractVector{Float64}
 x::A;y::B;z::C;first::Int32
end
Base.size(::TileVector)=(384,)
Base.IndexStyle(::Type{<:TileVector})=Base.IndexLinear()
@inline function Base.getindex(t::TileVector,index::Integer)
 i=Int32(index)-Int32(1);row=t.first+div(i,Int32(3))+Int32(1);axis=rem(i,Int32(3))
 @inbounds return axis==0 ? t.x[row] : axis==1 ? t.y[row] : t.z[row]
end
struct ZeroWords <: AbstractVector{UInt32} end
Base.size(::ZeroWords)=(2048,)
Base.IndexStyle(::Type{ZeroWords})=Base.IndexLinear()
@inline Base.getindex(::ZeroWords,::Integer)=UInt32(0)
function sym_entry!(i,j,x,y,z,out)
 gx=Int32(i)-Int32(1);lx=rem(gx,Int32(128));bx=div(gx,Int32(128));by=Int32(j)-Int32(1)
 if bx<by
  ignored_store!(out,gx,by)
 else
  tile=JACC.shared(TileVector(x,y,z,by*Int32(128)))
  @inbounds acf_store!(Val(true),out,tile,x[gx+Int32(1)],y[gx+Int32(1)],z[gx+Int32(1)],lx,gx,bx,by)
 end
 nothing
end
function cross_entry!(i,j,x1,y1,z1,x2,y2,z2,out)
 gx=Int32(i)-Int32(1);lx=rem(gx,Int32(128));bx=div(gx,Int32(128));by=Int32(j)-Int32(1)
 tile=JACC.shared(TileVector(x1,y1,z1,by*Int32(128)))
 @inbounds acf_store!(Val(false),out,tile,x2[gx+Int32(1)],y2[gx+Int32(1)],z2[gx+Int32(1)],lx,gx,bx,by)
 nothing
end
function hist_entry!(i,out,src,size::Int32)
 index=Int32(i)-Int32(1);lx=rem(index,Int32(128));bx=div(index,Int32(128))
 words=JACC.shared(ZeroWords());hist=reinterpret(UInt8,words)
 histogram_count!(hist,src,size,lx,bx);JACC.sync_workgroup();histogram_store!(out,hist,lx,bx);nothing
end
function merge_entry!(i,data,blocks::Int32)
 index=Int32(i)-Int32(1);merge_store!(data,blocks,rem(index,Int32(64)),div(index,Int32(64)));nothing
end
function prepare_dispatch(c)
 sx,sy,sz=c.input1;rx,ry,rz=c.input2;N=(16384,128);ext=Base.get_extension(JACC,:CUDAExt)
 # Pinned JACC backend introspection obtains constants from ACTUAL JACC entries.
 sk=ext._kernel_args(ext.BlockIndexerBasic(),N,sym_entry!,sx.d,sy.d,sz.d,c.packed.d)
 rk=ext._kernel_args(ext.BlockIndexerBasic(),N,cross_entry!,sx.d,sy.d,sz.d,rx.d,ry.d,rz.d,c.packed.d)
 sc,_=ext._kernel_maxshmem(ext._parallel_for_cuda_MN,sk,nothing);rc,_=ext._kernel_maxshmem(ext._parallel_for_cuda_MN,rk,nothing)
 symbols=(CUDA.CuGlobal{NTuple{63,Float64}}(sc.fun.mod,"tpacf_study_bins"),CUDA.CuGlobal{NTuple{63,Float64}}(rc.fun.mod,"tpacf_study_bins"))
 println("TPACF_PTX_BEGIN_SYM");CUDA.code_ptx(stdout,ext._parallel_for_cuda_MN,Tuple{Core.Typeof.(sk)...};kernel=true,dump_module=true);println("TPACF_PTX_END_SYM")
 ss=JACC.launch_spec(threads=(128,1),blocks=(128,128),shmem_size=3072,sync=false)
 rs=JACC.launch_spec(threads=(128,1),blocks=(128,128),shmem_size=3072,sync=false)
 symbols,(()->JACC.parallel_for(ss,N,sym_entry!,sx.d,sy.d,sz.d,c.packed.d)),(()->JACC.parallel_for(rs,N,cross_entry!,sx.d,sy.d,sz.d,rx.d,ry.d,rz.d,c.packed.d))
end
function hist_launch!(c,start,words,blocks)
 src=view(c.packed.d,start+1:length(c.packed.d));spec=JACC.launch_spec(threads=128,blocks=blocks,shmem_size=8192,sync=false)
 JACC.parallel_for(spec,blocks*128,hist_entry!,c.histo.d,src,Int32(words));nothing
end
function merge_launch!(c,blocks)
 spec=JACC.launch_spec(threads=64,blocks=32,shmem_size=0,sync=false);JACC.parallel_for(spec,2048,merge_entry!,c.histo.d,Int32(blocks));nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study()
