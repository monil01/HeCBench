using CUDA,KernelAbstractions
const MODEL="ka"
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function sym_entry!(x,y,z,out)
 ix=@index(Global,NTuple);localindex=@index(Local,NTuple);lx=Int32(localindex[1])-Int32(1);gx=Int32(ix[1])-Int32(1);bx=div(gx,Int32(128));by=Int32(ix[2])-Int32(1)
 tile=@localmem(Float64,(384,))
 if bx<by
  ignored_store!(out,gx,by)
 else
  row=by*Int32(128)+lx;at=Int32(3)*lx
  @inbounds begin tile[at+Int32(1)]=x[row+Int32(1)];tile[at+Int32(2)]=y[row+Int32(1)];tile[at+Int32(3)]=z[row+Int32(1)];end
  @synchronize
  @inbounds acf_store!(Val(true),out,tile,x[gx+Int32(1)],y[gx+Int32(1)],z[gx+Int32(1)],lx,gx,bx,by)
 end
end
@kernel function cross_entry!(x1,y1,z1,x2,y2,z2,out)
 ix=@index(Global,NTuple);localindex=@index(Local,NTuple);lx=Int32(localindex[1])-Int32(1);gx=Int32(ix[1])-Int32(1);bx=div(gx,Int32(128));by=Int32(ix[2])-Int32(1);row=by*Int32(128)+lx;at=Int32(3)*lx
 tile=@localmem(Float64,(384,))
 @inbounds begin tile[at+Int32(1)]=x1[row+Int32(1)];tile[at+Int32(2)]=y1[row+Int32(1)];tile[at+Int32(3)]=z1[row+Int32(1)];end
 @synchronize
 @inbounds acf_store!(Val(false),out,tile,x2[gx+Int32(1)],y2[gx+Int32(1)],z2[gx+Int32(1)],lx,gx,bx,by)
end
@kernel function hist_entry!(out,src,size::Int32)
 lx=Int32(@index(Local,Linear))-Int32(1);bx=Int32(@index(Group,Linear))-Int32(1)
 words=@localmem(UInt32,(2048,));zero_hist!(words,lx);@synchronize
 hist=reinterpret(UInt8,words);histogram_count!(hist,src,size,lx,bx);@synchronize
 histogram_store!(out,hist,lx,bx)
end
@kernel function merge_entry!(data,blocks::Int32)
 merge_store!(data,blocks,Int32(@index(Local,Linear))-Int32(1),Int32(@index(Group,Linear))-Int32(1))
end
function prepare_dispatch(c)
 sx,sy,sz=c.input1;rx,ry,rz=c.input2;N=(16384,128)
 sym=sym_entry!(BACKEND,(128,1));cross=cross_entry!(BACKEND,(128,1))
 ndr,wgs,iter,dynamic=KernelAbstractions.launch_config(sym,N,nothing);sctx=KernelAbstractions.mkcontext(sym,ndr,iter)
 ndr,wgs,iter,dynamic=KernelAbstractions.launch_config(cross,N,nothing);cctx=KernelAbstractions.mkcontext(cross,ndr,iter)
 # Compile actual KA entry for module introspection only; launches below use KA.
 sc=CUDA.@cuda launch=false always_inline=BACKEND.always_inline maxthreads=128 sym.f(sctx,sx.d,sy.d,sz.d,c.packed.d)
 rc=CUDA.@cuda launch=false always_inline=BACKEND.always_inline maxthreads=128 cross.f(cctx,sx.d,sy.d,sz.d,rx.d,ry.d,rz.d,c.packed.d)
 symbols=(CUDA.CuGlobal{NTuple{63,Float64}}(sc.fun.mod,"tpacf_study_bins"),CUDA.CuGlobal{NTuple{63,Float64}}(rc.fun.mod,"tpacf_study_bins"))
 println("TPACF_PTX_BEGIN_SYM");args=(sctx,sx.d,sy.d,sz.d,c.packed.d);CUDA.code_ptx(stdout,sym.f,Tuple{map(a->typeof(cudaconvert(a)),args)...};kernel=true,dump_module=true,maxthreads=128,always_inline=BACKEND.always_inline);println("TPACF_PTX_END_SYM")
 symbols,(()->sym(sx.d,sy.d,sz.d,c.packed.d;ndrange=N)),(()->cross(sx.d,sy.d,sz.d,rx.d,ry.d,rz.d,c.packed.d;ndrange=N))
end
function hist_launch!(c,start,words,blocks)
 src=view(c.packed.d,start+1:length(c.packed.d));hist_entry!(BACKEND,128)(c.histo.d,src,Int32(words);ndrange=blocks*128);nothing
end
function merge_launch!(c,blocks)
 merge_entry!(BACKEND,64)(c.histo.d,Int32(blocks);ndrange=2048);nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study()
