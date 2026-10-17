using CUDA
const THREADS=128
@inline wih_offset(l,g,h,k,H,M)=(((l*Int32(3)+g)*H+h)*M+k)+Int32(1)
@inline whh_offset(l,g,h,k,H)=(((l*Int32(3)+g)*H+h)*H+k)+Int32(1)
@inline bias_offset(l,g,h,H)=((l*Int32(3)+g)*H+h)+Int32(1)
@inline function gru_body!(tx,b,tile,x,hprev,wi,wh,bi,bh,out,hnext,output_t,layer,B,I,H,M,write_output)
 lane=mod(tx,Int32(32));h=tile*Int32(4)+tx÷Int32(32)
 if b>=B||h>=H;return;end
 ig0=0f0;ig1=0f0;ig2=0f0;hg0=0f0;hg1=0f0;hg2=0f0
 for gate in Int32(0):Int32(2)
  insum=0f0;hsum=0f0
  for k in lane:Int32(32):I-Int32(1)
   @inbounds insum+=x[b*I+k+Int32(1)]*wi[wih_offset(layer,gate,h,k,H,M)]
  end
  for k in lane:Int32(32):H-Int32(1)
   @inbounds hsum+=hprev[b*H+k+Int32(1)]*wh[whh_offset(layer,gate,h,k,H)]
  end
  for offset in (16,8,4,2,1)
   insum+=CUDA.shfl_down_sync(UInt32(0xffffffff),insum,offset)
   hsum+=CUDA.shfl_down_sync(UInt32(0xffffffff),hsum,offset)
  end
  if lane==0
   @inbounds a=insum+bi[bias_offset(layer,gate,h,H)];c=hsum+bh[bias_offset(layer,gate,h,H)]
   if gate==0;ig0=a;hg0=c;elseif gate==1;ig1=a;hg1=c;else;ig2=a;hg2=c;end
  end
 end
 if lane==0
  idx=b*H+h+Int32(1);@inbounds hx=hprev[idx]
  reset=1f0/(1f0+exp(-(ig0+hg0)));update=1f0/(1f0+exp(-(ig1+hg1)));candidate=tanh(ig2+reset*hg2);ht=candidate+update*(hx-candidate)
  @inbounds hnext[idx]=ht;out[idx]=ht
  if write_output!=0;@inbounds output_t[idx]=ht;end
 end
 nothing
end
