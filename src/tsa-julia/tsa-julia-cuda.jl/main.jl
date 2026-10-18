using CUDA
include("../study_common.jl")
function tsa_kernel(a::T,b,w,h,real,imag,result_r,result_i,::Val{BY},gx) where {T,BY}
 tx=Int32(threadIdx().x);ty=(tx-1)÷Int32(16);tx=(tx-1)%Int32(16)
 block=Int32(blockIdx().x)-1;bx=block%gx;by=block÷gx
 rl=CuStaticSharedArray(T,16BY);im=CuStaticSharedArray(T,16BY)

 px=bx*Int32(10)+tx-Int32(3);py=by*Int32(BY-8)+ty-Int32(4)
 if 0<=px<w
  for part in 0:(BY÷16-1)
   y=py+Int32(16part)
   if 0<=y<h
    @inbounds rl[(ty+Int32(16part))*Int32(16)+tx+Int32(1)]=real[y*w+px+Int32(1)];im[(ty+Int32(16part))*Int32(16)+tx+Int32(1)]=imag[y*w+px+Int32(1)]
   end
  end
 end
 sync_threads()
 sx=tx;sy=Int32(2)*ty+Int32(1)-tx%Int32(2);checker_y=by*Int32(BY-8)+sy-Int32(4)
 cr=MVector{BY÷32,T}(undef);ci=MVector{BY÷32,T}(undef)
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1]=rl[(sy+Int32(32part))*Int32(16)+sx+Int32(1)];ci[part+1]=im[(sy+Int32(32part))*Int32(16)+sx+Int32(1)]
 end
  for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),checker_y+Int32(32part),w,h,rl,im,Val(BY),Val(1),Val(0))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),px,w,h,rl,im,Val(BY),Val(0),Val(0))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),checker_y+Int32(32part),w,h,rl,im,Val(BY),Val(1),Val(1))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),px,w,h,rl,im,Val(BY),Val(0),Val(1))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),px,w,h,rl,im,Val(BY),Val(0),Val(1))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),checker_y+Int32(32part),w,h,rl,im,Val(BY),Val(1),Val(1))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),px,w,h,rl,im,Val(BY),Val(0),Val(0))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds cr[part+1],ci[part+1]=tsa_pair(a,b,cr[part+1],ci[part+1],sx,sy+Int32(32part),checker_y+Int32(32part),w,h,rl,im,Val(BY),Val(1),Val(0))
 end
 sync_threads()
 for part in 0:(BY÷32-1)
  @inbounds rl[(sy+Int32(32part))*Int32(16)+sx+Int32(1)]=cr[part+1];im[(sy+Int32(32part))*Int32(16)+sx+Int32(1)]=ci[part+1]
 end
 sync_threads()
 sx=tx+Int32(3);sy=ty+Int32(4);px+=Int32(3);py+=Int32(4)
 if sx<Int32(13) && px<w
  for part in 0:(BY÷16-1)
   y=py+Int32(16part)
   if sy+Int32(16part)<Int32(BY-4) && y<h
    @inbounds result_r[y*w+px+Int32(1)]=rl[(sy+Int32(16part))*Int32(16)+sx+Int32(1)];result_i[y*w+px+Int32(1)]=im[(sy+Int32(16part))*Int32(16)+sx+Int32(1)]
   end
  end
 end
 return
end
function factory(a,b,w,h,rs,ims)
 BY=sizeof(eltype(rs[1]))==8 ? 32 : 96;gx=cld(w,Int32(10));gy=cld(h,Int32(BY-8))
 sense->(@cuda threads=256 blocks=gx*gy tsa_kernel(a,b,w,h,rs[sense],ims[sense],rs[3-sense],ims[3-sense],Val(BY),gx))
end
tsa_study(factory)
