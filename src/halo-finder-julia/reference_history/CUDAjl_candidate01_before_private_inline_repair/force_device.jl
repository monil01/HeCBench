# Julia mathematical translation of unchanged source Step10 tile algorithm.
# LANL notices and original source retained in reference_history/RCBForceTree.cxx.
using CUDA,StaticArrays
@inline halo_rsqrt_rn(x::Float32)=ccall("extern __nv_frsqrt_rn",llvmcall,Float32,(Float32,),x)
@inline function halo_load(::Val{4},a,tile,count)
    p=reinterpret(Core.LLVMPtr{NTuple{4,VecElement{Float32}},CUDA.AS.Global},pointer(a,Int32(4)*tile+Int32(1)))
    v=CUDA.unsafe_cached_load(p,1,Val(16))
    SVector{4,Float32}(v[1].value,v[2].value,v[3].value,v[4].value)
end
@inline function halo_load(::Val{1},a,index,count)
    value=index<count ? CUDA.unsafe_cached_load(pointer(a),index+Int32(1),Val(4)) : 0f0
    SVector{1,Float32}(value)
end
@inline function halo_accumulate_one(x,y,z,x1,y1,z1,mass,ax,ay,az,soft2,cut2)
    dx=x1-x;dy=y1-y;dz=z1-z
    r2=muladd(dz,dz,muladd(dx,dx,dy*dy))
    v=r2+soft2;v3=(v*v)*v
    f=halo_rsqrt_rn(v3)*(mass*Float32(r2<cut2&&r2>0f0))
    return muladd(f,dx,ax),muladd(f,dy,ay),muladd(f,dz,az)
end
@generated function halo_compute!(::Val{TX},::Val{TY},ix,iy,iz,jx,jy,jz,jm,ax,ay,az,soft2,cut2) where {TX,TY}
    code=Expr(:block)
    for i in 1:TY,j in 1:TX
        push!(code.args,:(@inbounds ax[$i],ay[$i],az[$i]=halo_accumulate_one(ix[$i],iy[$i],iz[$i],jx[$j],jy[$j],jz[$j],jm[$j],ax[$i],ay[$i],az[$i],soft2,cut2)))
    end
    push!(code.args,:(nothing));code
end
@inline function halo_warp_update!(output,index,value)
    value+=CUDA.shfl_down_sync(UInt32(0xffffffff),value,16)
    value+=CUDA.shfl_down_sync(UInt32(0xffffffff),value,8)
    value+=CUDA.shfl_down_sync(UInt32(0xffffffff),value,4)
    value+=CUDA.shfl_down_sync(UInt32(0xffffffff),value,2)
    value+=CUDA.shfl_down_sync(UInt32(0xffffffff),value,1)
    if rem(Int32(CUDA.threadIdx().x)-Int32(1),Int32(32))==0
        CUDA.atomic_add!(pointer(output,index+Int32(1)),value)
    end
    nothing
end
@generated function halo_apply!(::Val{TY},tile,count,fcoeff,ax,ay,az,c) where TY
    code=Expr(:block)
    for i in 1:TY
        push!(code.args,quote
            index=Int32($TY)*tile+Int32($(i-1))
            if index<count
                @inbounds halo_warp_update!(c.vx,index,fcoeff*ax[$i])
                @inbounds halo_warp_update!(c.vy,index,fcoeff*ay[$i])
                @inbounds halo_warp_update!(c.vz,index,fcoeff*az[$i])
            end
        end)
    end
    push!(code.args,:(nothing));code
end
@inline function halo_step(c,count,count1,cut2,soft2,fcoeff,xidx,yidx,xstride,ystride)
    ni=div(count,Int32(4));nj=div(count1,Int32(4))
    for i in yidx:ystride:ni-Int32(1)
        ax=MVector{4,Float32}(0f0,0f0,0f0,0f0);ay=MVector{4,Float32}(0f0,0f0,0f0,0f0);az=MVector{4,Float32}(0f0,0f0,0f0,0f0)
        ix=halo_load(Val(4),c.x,i,count);iy=halo_load(Val(4),c.y,i,count);iz=halo_load(Val(4),c.z,i,count)
        for j in xidx:xstride:nj-Int32(1)
            jx=halo_load(Val(4),c.x1,j,count1);jy=halo_load(Val(4),c.y1,j,count1);jz=halo_load(Val(4),c.z1,j,count1);jm=halo_load(Val(4),c.mass1,j,count1)
            halo_compute!(Val(4),Val(4),ix,iy,iz,jx,jy,jz,jm,ax,ay,az,soft2,cut2)
        end
        for j in nj*Int32(4)+xidx:xstride:count1-Int32(1)
            jx=halo_load(Val(1),c.x1,j,count1);jy=halo_load(Val(1),c.y1,j,count1);jz=halo_load(Val(1),c.z1,j,count1);jm=halo_load(Val(1),c.mass1,j,count1)
            halo_compute!(Val(1),Val(4),ix,iy,iz,jx,jy,jz,jm,ax,ay,az,soft2,cut2)
        end
        halo_apply!(Val(4),i,count,fcoeff,ax,ay,az,c)
    end
    for i in yidx:ystride:count-ni*Int32(4)-Int32(1)
        k=i+ni*Int32(4)
        ax=MVector{1,Float32}(0f0);ay=MVector{1,Float32}(0f0);az=MVector{1,Float32}(0f0)
        ix=halo_load(Val(1),c.x,k,count);iy=halo_load(Val(1),c.y,k,count);iz=halo_load(Val(1),c.z,k,count)
        for j in xidx:xstride:nj-Int32(1)
            jx=halo_load(Val(4),c.x1,j,count1);jy=halo_load(Val(4),c.y1,j,count1);jz=halo_load(Val(4),c.z1,j,count1);jm=halo_load(Val(4),c.mass1,j,count1)
            halo_compute!(Val(4),Val(1),ix,iy,iz,jx,jy,jz,jm,ax,ay,az,soft2,cut2)
        end
        for j in nj*Int32(4)+xidx:xstride:count1-Int32(1)
            jx=halo_load(Val(1),c.x1,j,count1);jy=halo_load(Val(1),c.y1,j,count1);jz=halo_load(Val(1),c.z1,j,count1);jm=halo_load(Val(1),c.mass1,j,count1)
            halo_compute!(Val(1),Val(1),ix,iy,iz,jx,jy,jz,jm,ax,ay,az,soft2,cut2)
        end
        halo_apply!(Val(1),k,count,fcoeff,ax,ay,az,c)
    end
    nothing
end
