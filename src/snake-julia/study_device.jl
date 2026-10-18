@inline sk_lsl(x::UInt32,s)=0<s<32 ? x<<s : x
@inline sk_lsr(x::UInt32,s)=0<s<32 ? x>>s : x
@inline function snake_filter(A,B,base,T)
 reads=ntuple(k->@inbounds(A[base+k]),Val(8));refs=ntuple(k->@inbounds(B[base+k]),Val(8))
 j=Int32(0);shift=Int32(0);glob=Int32(0);errs=Int32(0);one=UInt32(1)
 while j<7&&glob<200
  ref1=sk_lsl(refs[j+1],shift);ref2=sk_lsr(refs[j+2],32-shift);ref=ref1|ref2
  best=leading_zeros(xor(sk_lsl(reads[j+1],shift)|sk_lsr(reads[j+2],32-shift),ref))
  for e=Int32(1):T
   corner=UInt32(0)
   if j==0&&shift-2*e<0
    read=sk_lsr(reads[j+1],2*e-shift)
    for c=Int32(0):Int32(2*e-shift-1);corner|=sk_lsl(one,31-c);end
   elseif shift-2*e<0
    read=sk_lsl(reads[j],32-(2*e-shift))|sk_lsr(reads[j+1],2*e-shift)
   else
    read=sk_lsl(reads[j+1],shift-2*e)|sk_lsr(reads[j+2],32-(shift-2*e))
   end
   best=max(best,leading_zeros(xor(read,ref)|corner))
  end
  for e=Int32(1):T
   corner=UInt32(0)
   if j<5
    if shift+2*e<32
     read=sk_lsl(reads[j+1],shift+2*e)|sk_lsr(reads[j+2],32-(shift+2*e))
    else
     read=sk_lsl(reads[j+2],(shift+2*e)%32)|sk_lsr(reads[j+3],32-(shift+2*e)%32)
    end
   else
    read=sk_lsl(reads[j+1],shift+2*e)|sk_lsr(reads[j+2],32-(shift+2*e))
    if glob+32>200
     for c=Int32(glob+32-200):Int32(glob+32-200+2*e-1);corner|=sk_lsl(one,c);end
    elseif glob+32>=200-2*e
     for c=Int32(0):Int32(2*e-1);corner|=sk_lsl(one,c);end
    end
   end
   best=max(best,leading_zeros(xor(read,ref)|corner))
  end
  z=Int32(div(best,2)*2)
  if j==6&&z>=8;break;end
  if(div(z,2)<16&&j<5)||(j==6&&div(z,2)<4);errs+=Int32(1);end
  if errs>T;return Int32(0);end
  if shift+z+2>=32;j+=Int32(1);end
  if z==32;glob+=z
  else;shift=Int32((shift+z+2)%32);glob+=z+Int32(2)
  end
 end
 Int32(1)
end
