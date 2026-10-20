using CUDA
const MODEL="cuda_jl"
include(joinpath(@__DIR__,"..","study_device.jl"))
function sym_entry!(x,y,z,out)
 lx=Int32(threadIdx().x)-Int32(1);bx=Int32(blockIdx().x)-Int32(1);by=Int32(blockIdx().y)-Int32(1);gx=bx*Int32(128)+lx
 tile=CUDA.@cuDynamicSharedMem(Float64,384)
 if bx<by
  ignored_store!(out,gx,by)
 else
  row=by*Int32(128)+lx;at=Int32(3)*lx
  @inbounds begin tile[at+Int32(1)]=x[row+Int32(1)];tile[at+Int32(2)]=y[row+Int32(1)];tile[at+Int32(3)]=z[row+Int32(1)];end
  CUDA.sync_threads()
  @inbounds acf_store!(Val(true),out,tile,x[gx+Int32(1)],y[gx+Int32(1)],z[gx+Int32(1)],lx,gx,bx,by)
 end
 nothing
end
function cross_entry!(x1,y1,z1,x2,y2,z2,out)
 lx=Int32(threadIdx().x)-Int32(1);bx=Int32(blockIdx().x)-Int32(1);by=Int32(blockIdx().y)-Int32(1);gx=bx*Int32(128)+lx;row=by*Int32(128)+lx;at=Int32(3)*lx
 tile=CUDA.@cuDynamicSharedMem(Float64,384)
 @inbounds begin tile[at+Int32(1)]=x1[row+Int32(1)];tile[at+Int32(2)]=y1[row+Int32(1)];tile[at+Int32(3)]=z1[row+Int32(1)];end
 CUDA.sync_threads()
 @inbounds acf_store!(Val(false),out,tile,x2[gx+Int32(1)],y2[gx+Int32(1)],z2[gx+Int32(1)],lx,gx,bx,by)
 nothing
end
function hist_entry!(out,src,size::Int32)
 lx=Int32(threadIdx().x)-Int32(1);bx=Int32(blockIdx().x)-Int32(1)
 words=CUDA.@cuStaticSharedMem(UInt32,2048);zero_hist!(words,lx);CUDA.sync_threads();hist=reinterpret(UInt8,words)
 histogram_count!(hist,src,size,lx,bx);CUDA.sync_threads();histogram_store!(out,hist,lx,bx);nothing
end
function merge_entry!(data,blocks::Int32)
 merge_store!(data,blocks,Int32(threadIdx().x)-Int32(1),Int32(blockIdx().x)-Int32(1));nothing
end
function prepare_dispatch(c)
 sx,sy,sz=c.input1;rx,ry,rz=c.input2
 sym=CUDA.@cuda launch=false sym_entry!(sx.d,sy.d,sz.d,c.packed.d)
 cross=CUDA.@cuda launch=false cross_entry!(sx.d,sy.d,sz.d,rx.d,ry.d,rz.d,c.packed.d)
 symbols=(CUDA.CuGlobal{NTuple{63,Float64}}(sym.fun.mod,"tpacf_study_bins"),CUDA.CuGlobal{NTuple{63,Float64}}(cross.fun.mod,"tpacf_study_bins"))
 println("TPACF_PTX_BEGIN_SYM");CUDA.code_ptx(stdout,sym_entry!,Tuple{typeof(cudaconvert(sx.d)),typeof(cudaconvert(sy.d)),typeof(cudaconvert(sz.d)),typeof(cudaconvert(c.packed.d))};kernel=true,dump_module=true);println("TPACF_PTX_END_SYM")
 symbols,(()->sym(sx.d,sy.d,sz.d,c.packed.d;threads=128,blocks=(128,128),shmem=3072)),(()->cross(sx.d,sy.d,sz.d,rx.d,ry.d,rz.d,c.packed.d;threads=128,blocks=(128,128),shmem=3072))
end
function hist_launch!(c,start,words,blocks)
 src=view(c.packed.d,start+1:length(c.packed.d));CUDA.@cuda threads=128 blocks=blocks hist_entry!(c.histo.d,src,Int32(words));nothing
end
function merge_launch!(c,blocks)
 CUDA.@cuda threads=64 blocks=32 merge_entry!(c.histo.d,Int32(blocks));nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study()
