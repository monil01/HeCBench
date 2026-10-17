using CUDA
@inline function quad_digits(quad::NTuple{4,UInt32},shift::UInt32)
 (Int32((quad[1]>>shift)&UInt32(15))+Int32(1),Int32((quad[2]>>shift)&UInt32(15))+Int32(1),Int32((quad[3]>>shift)&UInt32(15))+Int32(1),Int32((quad[4]>>shift)&UInt32(15))+Int32(1))
end
function reduce_ref!(input,sums,n,shift)
 lid=threadIdx().x-Int32(1);group=blockIdx().x-Int32(1)
 tile=CuStaticSharedArray(UInt32,256)
 region=(n÷Int32(4))÷Int32(64)*Int32(4);start=group*region;stop=group==Int32(63) ? n : start+region
 histogram=ntuple(_->UInt32(0),Val(16));i=start+lid
 while i<stop
  @inbounds digit=Int32((input[i+Int32(1)]>>shift)&UInt32(15))+Int32(1)
  histogram=Base.setindex(histogram,histogram[digit]+UInt32(1),digit);i+=Int32(256)
 end
 for digit in Int32(0):Int32(15)
  @inbounds tile[lid+Int32(1)]=histogram[digit+Int32(1)]
  sync_threads()
  for stride in (128,64,32,16,8,4,2,1)
   if lid<stride;@inbounds tile[lid+Int32(1)]+=tile[lid+stride+Int32(1)];end
   sync_threads()
  end
  if lid==0;@inbounds sums[digit*Int32(64)+group+Int32(1)]=tile[1];end
 end
 return
end
function top_ref!(sums)
 lid=threadIdx().x-Int32(1)
 tile=CuStaticSharedArray(UInt32,513)
 if lid==0;tile[513]=0;end
 sync_threads()
 for digit in Int32(0):Int32(15)
  value=lid<Int32(64) ? sums[digit*Int32(64)+lid+Int32(1)] : UInt32(0)
  idx=lid+Int32(256);tile[lid+Int32(1)]=0;tile[idx+Int32(1)]=value
  sync_threads()
  for offset in (1,2,4,8,16,32,64,128)
   @inbounds previous=tile[idx-offset+Int32(1)]
   sync_threads()
   @inbounds tile[idx+Int32(1)]+=previous
   sync_threads()
  end
  @inbounds result=tile[idx]
  if lid<Int32(64);@inbounds sums[digit*Int32(64)+lid+Int32(1)]=result+tile[513];end
  sync_threads()
  if lid==Int32(63);tile[513]+=result+value;end
  sync_threads()
 end
 return
end
function bottom_ref!(output,input,sums,n,shift)
 lid=threadIdx().x-Int32(1);group=blockIdx().x-Int32(1)
 tile=CuStaticSharedArray(UInt32,544)
 region=(n÷Int32(4))÷Int32(64);start=group*region;stop=group==Int32(63) ? n÷Int32(4) : start+region
 i=start+lid;window=start
 if lid<Int32(16)
  tile[529+lid]=0;tile[513+lid]=sums[lid*Int32(64)+group+Int32(1)]
 end
 sync_threads()
 while window<stop
  histogram=ntuple(_->UInt32(0),Val(16));quad=(UInt32(0),UInt32(0),UInt32(0),UInt32(0));keys=(Int32(0),Int32(0),Int32(0),Int32(0))
  if i<stop
   @inbounds quad=(input[i*Int32(4)+Int32(1)],input[i*Int32(4)+Int32(2)],input[i*Int32(4)+Int32(3)],input[i*Int32(4)+Int32(4)])
   keys=quad_digits(quad,shift)
   for k in 1:4;digit=keys[k];histogram=Base.setindex(histogram,histogram[digit]+UInt32(1),digit);end
  end
  for digit in Int32(1):Int32(16)
   idx=lid+Int32(256);tile[lid+Int32(1)]=0;tile[idx+Int32(1)]=histogram[digit]
   sync_threads()
   for offset in (1,2,4,8,16,32,64,128)
    @inbounds previous=tile[idx-offset+Int32(1)]
    sync_threads()
    @inbounds tile[idx+Int32(1)]+=previous
    sync_threads()
   end
   @inbounds histogram=Base.setindex(histogram,tile[idx],digit)
   sync_threads()
  end
  if i<stop
   for k in 1:4
    digit=keys[k];@inbounds address=histogram[digit]+tile[512+digit]+tile[528+digit];output[address+UInt32(1)]=quad[k]
    histogram=Base.setindex(histogram,histogram[digit]+UInt32(1),digit)
   end
  end
  sync_threads()
  if lid==Int32(255)
   for q in 1:16;tile[528+q]+=histogram[q];end
  end
  sync_threads();window+=Int32(256);i+=Int32(256)
 end
 return
end
