using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function score_study(factory)
 docs=64;vocab=16384;repeats=10;lines=sum(1+mod(d,4) for d in 0:docs-1)
 d1=fill(UInt32(0xffffffff),lines*64);d2=copy(d1);weights=zeros(UInt64,vocab);bloom=zeros(UInt32,16384);info=zeros(UInt64,docs)
 for t in 0:vocab-1
  if mod(t,5)==0
   weights[t+1]=(UInt64(mod(t,17))<<24)+UInt64(mod(t,31)+1)
   for h in (t>>5,t&0x7ffff);bloom[(h>>5)+1]|=UInt32(1)<<(h&31);end
  end
 end
 score=zeros(Int64,docs);expectedhi=zeros(Int64,lines);expectedlo=zeros(Int64,lines);start=0
 for d in 0:docs-1
  count=1+mod(d,4);len=128*count-7-mod(d,17);info[d+1]=(UInt64(start)<<32)|UInt64(start+count-1)
  for i in 0:cld(len,2)-1
   t=mod(17*i+13*d,vocab);f=UInt32(1+mod(11*i+d,254));d1[start*64+i+1]=(UInt32(t)<<8)|f
  end
  for i in 0:div(len,2)-1
   t=mod(29*i+19*d+7,vocab);f=UInt32(1+mod(7*i+d,254));d2[start*64+i+1]=(UInt32(t)<<8)|f
  end
  for line in start:start+count-1
   total=UInt64(0)
   for arr in (d1,d2),j in 0:63
    entry=arr[line*64+j+1];entry==UInt32(0xffffffff)&&continue
    total+=weights[Int(entry>>8)+1]*UInt64(entry&255)
   end
   expectedhi[line+1]=Int64(total>>32);expectedlo[line+1]=Int64(total&0xffffffff);score[d+1]+=Int64(total)
  end
  start+=count
 end
 a=CuArray(d1);z=CuArray(d2);w1=CuArray(weights);w2=CuArray(weights);bh=CuArray(bloom);di=CuArray(info);hi=CUDA.zeros(UInt32,lines);lo=CUDA.zeros(UInt32,lines);out=CUDA.zeros(UInt64,docs)
 launch! = factory(a,z,w1,w2,bh,hi,lo,di,out,lines)
 OwnedStudy.run_trials!(launch!,()->(fill!(out,0);fill!(hi,0);fill!(lo,0)),()->vcat(Int64.(Array(out)),Int64.(Array(hi)),Int64.(Array(lo))),vcat(score,expectedhi,expectedlo),repeats;atol=0.,rtol=0.)
end
