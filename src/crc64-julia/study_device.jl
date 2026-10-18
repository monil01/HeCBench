# // *****************************************************************************
# //                   Copyright (C) 2014, UChicago Argonne, LLC
# //                              All Rights Reserved
# // 	       High-Performance CRC64 Library (ANL-SF-14-095)
# //                    Hal Finkel, Argonne National Laboratory
# // 
# //                              OPEN SOURCE LICENSE
# // 
# // Under the terms of Contract No. DE-AC02-06CH11357 with UChicago Argonne, LLC,
# // the U.S. Government retains certain rights in this software.
# // 
# // Redistribution and use in source and binary forms, with or without
# // modification, are permitted provided that the following conditions are met:
# // 
# // 1. Redistributions of source code must retain the above copyright notice, this
# //    list of conditions and the following disclaimer.
# // 
# // 2. Redistributions in binary form must reproduce the above copyright notice,
# //    this list of conditions and the following disclaimer in the documentation
# //    and/or other materials provided with the distribution.
# // 
# // 3. Neither the names of UChicago Argonne, LLC or the Department of Energy nor
# //    the names of its contributors may be used to endorse or promote products
# //    derived from this software without specific prior written permission. 
# //  
# // *****************************************************************************
# //                                  DISCLAIMER
# // 
# // THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
# // 
# // NEITHER THE UNTED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
# // ENERGY, NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY
# // WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY
# // FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA,
# // APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
# // INFRINGE PRIVATELY OWNED RIGHTS.
# // 
# // *****************************************************************************
# 
using StaticArrays
@inline function crc_interleaved(data,words,start,n,table,inter)
 cs=MVector{5,UInt64}(typemax(UInt64),0,0,0,0);p=start;stop=start+n
 @inbounds begin
 while p<stop && (rem(p,4)!=0 || stop-p<20)
  idx=Int((cs[1] ⊻ UInt64(data[p+1])) & 0xff);p+=1;cs[1]=table[768+idx+1] ⊻ (cs[1]>>8)
 end
 if p==stop;return cs[1] ⊻ typemax(UInt64);end
 cry=UInt64(0);wi=div(p,4);input=MVector{5,UInt32}(words[wi+1],words[wi+2],words[wi+3],words[wi+4],words[wi+5]);p+=20
 while stop-p>=20
  cs[1] ⊻=cry
  for i in 1:5
   input[i] ⊻=UInt32(cs[i]&0xffffffff)
   if i<5;cs[i+1] ⊻=cs[i]>>32;else;cry=cs[i]>>32;end
   cs[i]=inter[Int(input[i]&0xff)+1];input[i]>>=8
  end
  for b in 1:2
   for i in 1:5;cs[i] ⊻=inter[256*b+Int(input[i]&0xff)+1];input[i]>>=8;end
  end
  for i in 1:5;cs[i] ⊻=inter[768+Int(input[i]&0xff)+1];input[i]=words[div(p,4)+i];end
  p+=20
 end
 cs[1] ⊻=cry
 for i in 1:5
  if i>1;cs[1] ⊻=cs[i];end
  input[i] ⊻=UInt32(cs[1]&0xffffffff);cs[1]>>=32
  for b in 0:2;cs[1] ⊻=table[256*b+Int(input[i]&0xff)+1];input[i]>>=8;end
  cs[1] ⊻=table[768+Int(input[i]&0xff)+1]
 end
 while p<stop
  idx=Int((cs[1] ⊻ UInt64(data[p+1]))&0xff);p+=1;cs[1]=table[768+idx+1] ⊻ (cs[1]>>8)
 end
 end
 cs[1] ⊻ typemax(UInt64)
end
@inline function crc_lane!(tid,sizes,checks,data,words,table,inter,n,nthreads)
 bpt=div(n,nthreads);start=bpt*tid;sz=tid==nthreads-1 ? n-start : bpt
 @inbounds sizes[tid+1]=UInt64(sz)
 @inbounds checks[tid+1]=crc_interleaved(data,words,start,sz,table,inter)
 nothing
end
function crc_multiply(a::UInt64,b::UInt64)
 if (a ⊻ (a-UInt64(1))) < (b ⊻ (b-UInt64(1)));a,b=b,a;end
 a==0&&return UInt64(0)
 r=UInt64(0);h=UInt64(1)<<63
 while a!=0
  if a&h!=0;r ⊻=b;a ⊻=h;end
  b=(b>>1) ⊻ (b&1!=0 ? UInt64(0xc96c5795d7870f42) : UInt64(0));a<<=1
 end
 r
end
function crc_combine(a::UInt64,b::UInt64,n,powers)
 v=UInt64(8n);x=UInt64(1)<<63;i=1
 while v!=0
  if v&1!=0;x=crc_multiply(x,powers[i]);end
  v>>=1;i+=1
 end
 b ⊻ crc_multiply(a,x)
end
