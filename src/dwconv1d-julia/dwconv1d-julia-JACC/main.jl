using CUDA
import JACC
JACC.@init_backend
const FORWARD_TEMPLATE=JACC.zeros(Float32,6912)
const BACKWARD_TEMPLATE=JACC.zeros(Float32,3840)
function forward!(xx,yy,w,k,x,eps,B::Int32,C::Int32,T::Int32,template)
 G=8
 sh=JACC.shared(template);ww=view(sh,1:768);kk=view(sh,769:6912)
 i=Int32(yy)-Int32(1);t=(Int32(mod(xx-1,T÷4)))*Int32(4);ti=t+T*i;tj=T*B*C÷Int32(G)
 for q in Int32(0):Int32(3)
  ww[t+q+1]=w[t+T*(i%C)+q+1]
  for j in Int32(0):Int32(G-1)
   kk[t+Int32(768)*j+q+1]=k[ti+tj*j+q+1]
  end
 end
 JACC.sync_workgroup()
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
 return
end
function backward!(xx,yy,w,k,g,gw,gk,B::Int32,C::Int32,T::Int32,template)
 G=2
 sh=JACC.shared(template);ww=view(sh,1:768);kk=view(sh,769:2304);gg=view(sh,2305:3840)
 i=Int32(yy)-Int32(1);t=(Int32(mod(xx-1,T÷4)))*Int32(4);ti=t+T*i;tj=T*B*C÷Int32(G)
 for q in Int32(0):Int32(3)
  ww[t+q+1]=w[t+T*(i%C)+q+1]
  for j in Int32(0):Int32(G-1)
   kk[t+Int32(768)*j+q+1]=k[ti+tj*j+q+1]
   gg[t+Int32(768)*j+q+1]=g[ti+tj*j+q+1]
  end
 end
 JACC.sync_workgroup()
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
 return
end
function launch(v,eps,B,C,T)
 w,k,g,x,gw,gk=v
 f=JACC.launch_spec(threads=(T÷4,1),blocks=(1,B*C÷8),shmem_size=27648,sync=false)
 b=JACC.launch_spec(threads=(T÷4,1),blocks=(1,B*C÷2),shmem_size=15360,sync=false)
 JACC.parallel_for(f,(T÷4,B*C÷8),forward!,w,k,x,eps,Int32(B),Int32(C),Int32(T),FORWARD_TEMPLATE)
 JACC.parallel_for(b,(T÷4,B*C÷2),backward!,w,k,g,gw,gk,Int32(B),Int32(C),Int32(T),BACKWARD_TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
