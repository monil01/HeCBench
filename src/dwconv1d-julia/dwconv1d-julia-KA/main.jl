using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
@kernel function forward!(w,k,x,eps,B::Int32,C::Int32,T::Int32)
 G=8
 ww=@localmem(Float32,(768,));kk=@localmem(Float32,(6144,))
 i=Int32(@index(Group,Cartesian)[2])-Int32(1);t=(Int32(@index(Local,Cartesian)[1])-Int32(1))*Int32(4);ti=t+T*i;tj=T*B*C÷Int32(G)
 for q in Int32(0):Int32(3)
  ww[t+q+1]=w[t+T*(i%C)+q+1]
  for j in Int32(0):Int32(G-1)
   kk[t+Int32(768)*j+q+1]=k[ti+tj*j+q+1]
  end
 end
 @synchronize
 for j in Int32(0):Int32(G-1)
  sx=eps;sy=eps;sz=eps;sw=eps
  for u in Int32(0):t
   q=Int32(768)*j;kv=kk[u+q+1];r=T-t+u-Int32(4)
   sx=muladd(ww[r+Int32(4)+Int32(0)],kv,sx);sy=muladd(ww[r+Int32(3)+Int32(0)],kv,sy);sz=muladd(ww[r+Int32(2)+Int32(0)],kv,sz);sw=muladd(ww[r+Int32(1)+Int32(0)],kv,sw)
  end
  q=Int32(768)*j;r=T-Int32(3);kv1=kk[q+t+Int32(2)];kv2=kk[q+t+Int32(3)];kv3=kk[q+t+Int32(4)]
  sy=muladd(ww[r+Int32(3)+Int32(0)],kv1,sy);sz=muladd(ww[r+Int32(2)+Int32(0)],kv1,sz);sz=muladd(ww[r+Int32(3)+Int32(0)],kv2,sz);sw=muladd(ww[r+Int32(1)+Int32(0)],kv1,sw);sw=muladd(ww[r+Int32(2)+Int32(0)],kv2,sw);sw=muladd(ww[r+Int32(3)+Int32(0)],kv3,sw)
  z=ti+tj*j+Int32(1);x[z]=sx;x[z+1]=sy;x[z+2]=sz;x[z+3]=sw
 end
end
@kernel function backward!(w,k,g,gw,gk,B::Int32,C::Int32,T::Int32)
 G=2
 ww=@localmem(Float32,(768,));kk=@localmem(Float32,(1536,));gg=@localmem(Float32,(1536,))
 i=Int32(@index(Group,Cartesian)[2])-Int32(1);t=(Int32(@index(Local,Cartesian)[1])-Int32(1))*Int32(4);ti=t+T*i;tj=T*B*C÷Int32(G)
 for q in Int32(0):Int32(3)
  ww[t+q+1]=w[t+T*(i%C)+q+1]
  for j in Int32(0):Int32(G-1)
   kk[t+Int32(768)*j+q+1]=k[ti+tj*j+q+1]
   gg[t+Int32(768)*j+q+1]=g[ti+tj*j+q+1]
  end
 end
 @synchronize
 for j in Int32(0):Int32(G-1)
  sx=0f0;sy=0f0;sz=0f0;sw=0f0
  for u in Int32(0):t
   q=Int32(768)*j;kv=kk[u+q+1];r=T-t+u-Int32(4)
   sx=muladd(gg[r+Int32(4)+q],kv,sx);sy=muladd(gg[r+Int32(3)+q],kv,sy);sz=muladd(gg[r+Int32(2)+q],kv,sz);sw=muladd(gg[r+Int32(1)+q],kv,sw)
  end
  q=Int32(768)*j;r=T-Int32(3);kv1=kk[q+t+Int32(2)];kv2=kk[q+t+Int32(3)];kv3=kk[q+t+Int32(4)]
  sy=muladd(gg[r+Int32(3)+q],kv1,sy);sz=muladd(gg[r+Int32(2)+q],kv1,sz);sz=muladd(gg[r+Int32(3)+q],kv2,sz);sw=muladd(gg[r+Int32(1)+q],kv1,sw);sw=muladd(gg[r+Int32(2)+q],kv2,sw);sw=muladd(gg[r+Int32(3)+q],kv3,sw)
  z=ti+tj*j+Int32(1);gw[z]=sx;gw[z+1]=sy;gw[z+2]=sz;gw[z+3]=sw
  sx=0f0;sy=0f0;sz=0f0;sw=0f0
  for u in t+Int32(3):T-Int32(1)
   wv=ww[u+1];r=q+T+t-u-Int32(1)
   sx=muladd(gg[r+1],wv,sx);sy=muladd(gg[r+2],wv,sy);sz=muladd(gg[r+3],wv,sz);sw=muladd(gg[r+4],wv,sw)
  end
  r=q+T-Int32(3)
  sx=muladd(gg[r+3],ww[t+1],sx);sx=muladd(gg[r+2],ww[t+2],sx);sx=muladd(gg[r+1],ww[t+3],sx);sy=muladd(gg[r+3],ww[t+2],sy);sy=muladd(gg[r+2],ww[t+3],sy);sz=muladd(gg[r+3],ww[t+3],sz)
  gk[z]=sx;gk[z+1]=sy;gk[z+2]=sz;gk[z+3]=sw
 end
end
function launch(v,eps,B,C,T)
 w,k,g,x,gw,gk=v
 forward!(BACKEND,(T÷4,1))(w,k,x,eps,Int32(B),Int32(C),Int32(T);ndrange=(T÷4,B*C÷8))
 backward!(BACKEND,(T÷4,1))(w,k,g,gw,gk,Int32(B),Int32(C),Int32(T);ndrange=(T÷4,B*C÷2))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
