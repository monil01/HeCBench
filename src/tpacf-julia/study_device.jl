# Source kernel arithmetic, not independent mathematical oracle.
include(joinpath(@__DIR__,"study_constant.jl"))
@inline function angular_byte(dot)
    @inbounds begin
    if dot < load_bound(Int32(30))
        return UInt32(124)
    elseif dot < load_bound(Int32(29))
        return UInt32(120)
    elseif dot < load_bound(Int32(28))
        return UInt32(116)
    elseif dot < load_bound(Int32(27))
        return UInt32(112)
    elseif dot < load_bound(Int32(26))
        return UInt32(108)
    elseif dot < load_bound(Int32(25))
        return UInt32(104)
    elseif dot < load_bound(Int32(24))
        return UInt32(100)
    elseif dot < load_bound(Int32(23))
        return UInt32(96)
    elseif dot < load_bound(Int32(22))
        return UInt32(92)
    elseif dot < load_bound(Int32(21))
        return UInt32(88)
    elseif dot < load_bound(Int32(20))
        return UInt32(84)
    elseif dot < load_bound(Int32(19))
        return UInt32(80)
    elseif dot < load_bound(Int32(18))
        return UInt32(76)
    elseif dot < load_bound(Int32(17))
        return UInt32(72)
    elseif dot < load_bound(Int32(16))
        return UInt32(68)
    elseif dot < load_bound(Int32(15))
        return UInt32(64)
    elseif dot < load_bound(Int32(14))
        return UInt32(60)
    elseif dot < load_bound(Int32(13))
        return UInt32(56)
    elseif dot < load_bound(Int32(12))
        return UInt32(52)
    elseif dot < load_bound(Int32(11))
        return UInt32(48)
    elseif dot < load_bound(Int32(10))
        return UInt32(44)
    elseif dot < load_bound(Int32(9))
        return UInt32(40)
    elseif dot < load_bound(Int32(8))
        return UInt32(36)
    elseif dot < load_bound(Int32(7))
        return UInt32(32)
    elseif dot < load_bound(Int32(6))
        return UInt32(28)
    elseif dot < load_bound(Int32(5))
        return UInt32(24)
    elseif dot < load_bound(Int32(4))
        return UInt32(20)
    elseif dot < load_bound(Int32(3))
        return UInt32(16)
    elseif dot < load_bound(Int32(2))
        return UInt32(12)
    elseif dot < load_bound(Int32(1))
        return UInt32(8)
    elseif dot < load_bound(Int32(0))
        return UInt32(4)
    else
        return UInt32(0)
    end
    end
end
@inline function ignored_store!(out,gx,by)
 base=by*Int32(128*4096)+gx
 @inbounds for i in Int32(0):Int32(4):Int32(124);out[base+i*Int32(4096)+Int32(1)]=UInt32(2088533116);end
 nothing
end
@inline function acf_store!(::Val{Sym},out,tile,vx,vy,vz,lx,gx,bx,by) where Sym
 base=by*Int32(128*4096)+gx
 @inbounds for i in Int32(0):Int32(4):Int32(124)
  packed=UInt32(0)
  for j in Int32(0):Int32(3)
   if Sym && bx==by && lx<=i+j
    byte=UInt32(124)
   else
    at=Int32(3)*(i+j);dot=vx*tile[at+Int32(1)]+vy*tile[at+Int32(2)]+vz*tile[at+Int32(3)];byte=angular_byte(dot)
   end
   packed+=byte<<(j*Int32(8))
  end
  out[base+i*Int32(4096)+Int32(1)]=packed
 end
 nothing
end
@inline function zero_hist!(words,lx)
 @inbounds for pos in lx:Int32(128):Int32(2047);words[pos+Int32(1)]=UInt32(0);end
 nothing
end
@inline function histogram_count!(hist,src,size,lx,bx)
 threadpos=(lx & ~Int32(63)) | ((lx & Int32(15))<<Int32(2)) | ((lx & Int32(48))>>Int32(4))
 start=bx*Int32(8064);amount=min(size-start,Int32(8064));pos=lx
 @inbounds while pos<amount
  word=src[start+pos+Int32(1)]
  hist[threadpos+Int32(128)*Int32((word>>Int32(2))&UInt32(63))+Int32(1)]+=UInt8(1)
  hist[threadpos+Int32(128)*Int32((word>>Int32(10))&UInt32(63))+Int32(1)]+=UInt8(1)
  hist[threadpos+Int32(128)*Int32((word>>Int32(18))&UInt32(63))+Int32(1)]+=UInt8(1)
  hist[threadpos+Int32(128)*Int32((word>>Int32(26))&UInt32(63))+Int32(1)]+=UInt8(1)
  pos+=Int32(128)
 end
 nothing
end
@inline function histogram_store!(out,hist,lx,bx)
 if lx<Int32(64)
  sum=UInt32(0);acc=(lx & Int32(15))*Int32(4);start=lx*Int32(128)
  @inbounds for i in Int32(0):Int32(127)
   sum+=UInt32(hist[start+acc+Int32(1)]);acc+=Int32(1);if acc==Int32(128);acc=Int32(0);end
  end
  @inbounds out[bx*Int32(64)+lx+Int32(1)]=sum
 end
 nothing
end
@inline function merge_store!(data,blocks,lx,bx)
 globalpos=bx*Int32(64)+lx;size=blocks*Int32(64);sum=UInt32(0);pos=globalpos
 @inbounds while pos<size;sum+=data[pos+Int32(1)];pos+=Int32(2048);end
 @inbounds data[globalpos+Int32(1)]=sum
 nothing
end
