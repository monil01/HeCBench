using CUDA
const EMPTY_KEY=typemax(UInt32)
const FULL_WARP=UInt32(0xffffffff)
@inline match_any64(mask::UInt32,value::UInt64)=ccall("llvm.nvvm.match.any.sync.i64",llvmcall,UInt32,(UInt32,UInt64),mask,value)
@inline function murmur_key(data,pos::Int32,len::Int32,capacity::Int32)
 h=UInt32(0x3FB0BB5F)⊻UInt32(len);i=Int32(0);remaining=len;m=UInt32(0x5bd1e995)
 while remaining>=Int32(4)
  @inbounds word=UInt32(data[pos+i+Int32(1)])|(UInt32(data[pos+i+Int32(2)])<<8)|(UInt32(data[pos+i+Int32(3)])<<16)|(UInt32(data[pos+i+Int32(4)])<<24)
  word*=m;word⊻=word>>24;word*=m;h*=m;h⊻=word;i+=Int32(4);remaining-=Int32(4)
 end
 if remaining>=Int32(3);@inbounds h⊻=UInt32(data[pos+i+Int32(3)])<<16;end
 if remaining>=Int32(2);@inbounds h⊻=UInt32(data[pos+i+Int32(2)])<<8;end
 if remaining>=Int32(1);@inbounds h⊻=UInt32(data[pos+i+Int32(1)]);h*=m;end
 h⊻=h>>13;h*=m;h⊻=h>>15
 h%UInt32(capacity)
end
@inline function equal_key(a,ap,b,bp,len)
 for i in Int32(0):len-Int32(1)
  @inbounds if a[ap+i+Int32(1)]!=b[bp+i+Int32(1)];return false;end
 end
 true
end
@inline function get_insert!(reads,pos,len,base,capacity,keylen,keypos,counts,exts,extcounts)
 bucket=Int32(murmur_key(reads,pos,len,capacity))
 while true
  slot=base+bucket+Int32(1)
  previous=CUDA.atomic_cas!(pointer(keylen,slot),EMPTY_KEY,UInt32(len))
  cohort=match_any64(CUDA.active_mask(),UInt64(slot))
  if previous==EMPTY_KEY
   @inbounds keypos[slot]=pos;exts[slot]=0;extcounts[slot]=0
   for c in Int32(0):Int32(7);@inbounds counts[(slot-Int32(1))*Int32(8)+c+Int32(1)]=0;end
  end
  CUDA.sync_warp(cohort)
  if previous==EMPTY_KEY||(@inbounds keylen[slot]==UInt32(len)&&equal_key(reads,keypos[slot],reads,pos,len));return slot;end
  bucket=(bucket+Int32(1))%capacity
 end
end
@inline function base_rating(low::UInt32,high::UInt32,depth::Float64)
 viable=max(.2*depth,2.);expected=max(.3*depth,2.)
 low==0&&return UInt32(0)
 low==1&&return UInt32(1)
 low<viable&&return UInt32(2)
 expected>low&&low>=viable&&high<viable&&return UInt32(3)
 expected>low&&low>=viable&&high>=viable&&return UInt32(4)
 low>=expected&&high<viable&&return UInt32(5)
 low>=expected&&viable<high&&high<expected&&return UInt32(6)
 UInt32(7)
end
@inline function base_record(counts,slot,channel,depth)
 base=(slot-Int32(1))*Int32(8)
 @inbounds low=counts[base+channel+Int32(1)];high=counts[base+channel+Int32(5)]
 basechar=(UInt8(65),UInt8(67),UInt8(71),UInt8(84))[channel+Int32(1)]
 (base_rating(low,high,depth),high,low,basechar)
end
@inline function higher_record(a,b)
 a[1]!=b[1]&&return a[1]>b[1]
 a[2]!=b[2]&&return a[2]>b[2]
 a[3]!=b[3]&&return a[3]>b[3]
 true
end
@inline function set_extension!(slot,counts,exts,extcounts,depth)
 items=(base_record(counts,slot,Int32(0),depth),base_record(counts,slot,Int32(1),depth),base_record(counts,slot,Int32(2),depth),base_record(counts,slot,Int32(3),depth))
 # Preserve native pairwise comparator/swap order, including its equal-record tie rule.
 for i in 1:4,j in 1:4
  if higher_record(items[i],items[j])
   a=items[i];b=items[j];items=Base.setindex(items,b,i);items=Base.setindex(items,a,j)
  end
 end
 top=items[1][1];runner=items[2][1];ext=UInt8(88)
 if top>0
  if top<=3
   runner==0&&(ext=items[1][4])
  elseif top<6
   runner<3&&(ext=items[1][4])
  elseif top==6
   runner<4&&(ext=items[1][4])
  else
   if runner<7
    ext=items[1][4]
   elseif items[3][1]==7||items[1][3]==items[2][3]
    ext=UInt8(70)
   elseif items[1][3]>items[2][3]
    ext=items[1][4]
   elseif items[2][3]>items[1][3]
    ext=items[2][4]
   end
  end
 end
 count=UInt32(0)
 for i in 1:4;if items[i][4]==ext;count=items[i][3];break;end;end
 @inbounds exts[slot]=ext;extcounts[slot]=count
 nothing
end
@inline function assembly_body!(group,lane,ctg_offsets,contigs,reads,quals,read_offsets,read_counts,prefix_ht,depth,keylen,keypos,counts,exts,extcounts,visited_len,visited_pos,visited_val,longest,temporary,walklens,kstart,maxwalk,maxmer,ncontigs,qualoffset)
 group>=ncontigs&&return
 @inbounds ctg_start=group==0 ? Int32(0) : Int32(ctg_offsets[group]);ctg_length=Int32(ctg_offsets[group+Int32(1)])-ctg_start
 @inbounds read_start_index=group==0 ? Int32(0) : Int32(read_counts[group]);nreads=Int32(read_counts[group+Int32(1)])-read_start_index
 @inbounds byte_start=read_start_index==0 ? Int32(0) : Int32(read_offsets[read_start_index])
 @inbounds table_base=group==0 ? Int32(0) : Int32(prefix_ht[group]);capacity=Int32(prefix_ht[group+Int32(1)])-table_base;local_depth=depth[group+Int32(1)]
 visited_base=group*maxwalk;temp_base=group*(maxwalk+maxmer);longest_base=group*maxwalk
 k=kstart;max_k=min(Int32(121),ctg_length);shift=Int32(0);longest_length=Int32(0)
 while k>=Int32(21)&&k<=max_k
  for bucket in lane:Int32(32):capacity-Int32(1);@inbounds keylen[table_base+bucket+Int32(1)]=EMPTY_KEY;end
  running_bytes=Int32(0)
  for read in Int32(0):nreads-Int32(1)
   read_index=read_start_index+read
   @inbounds endpoint=Int32(read_offsets[read_index+Int32(1)]);previous=read==0 ? byte_start : Int32(read_offsets[read_index]);read_length=endpoint-previous
   # Same native continue placement; named fixture has no reads shorter than k.
   if k>read_length;continue;end
   position=byte_start+running_bytes;num_mers=read_length-k
   for start in lane:Int32(32):num_mers-Int32(1)
    slot=get_insert!(reads,position+start,k,table_base,capacity,keylen,keypos,counts,exts,extcounts)
    ext_pos=position+start+k
    @inbounds ext=reads[ext_pos+Int32(1)]
    ext==UInt8(78)&&continue
    channel=ext==UInt8(65) ? Int32(0) : ext==UInt8(67) ? Int32(1) : ext==UInt8(71) ? Int32(2) : ext==UInt8(84) ? Int32(3) : Int32(-1)
    @inbounds quality=Int32(quals[ext_pos+Int32(1)])-qualoffset
    if channel>=0
     index=(slot-Int32(1))*Int32(8)+channel+Int32(1)
     if quality>=Int32(10);CUDA.@atomic counts[index]+=UInt32(1);end
     if quality>=Int32(20);CUDA.@atomic counts[index+Int32(4)]+=UInt32(1);end
    end
   end
   CUDA.sync_warp(FULL_WARP);running_bytes+=read_length
  end
  CUDA.sync_warp(FULL_WARP)
  for bucket in lane:Int32(32):capacity-Int32(1)
   slot=table_base+bucket+Int32(1)
   @inbounds if keylen[slot]!=EMPTY_KEY;set_extension!(slot,counts,exts,extcounts,local_depth);end
  end
  CUDA.sync_warp(FULL_WARP)
  for bucket in lane:Int32(32):maxwalk-Int32(1);@inbounds visited_len[visited_base+bucket+Int32(1)]=EMPTY_KEY;end
  if lane==0
   for i in Int32(0):k-Int32(1);@inbounds temporary[temp_base+i+Int32(1)]=contigs[ctg_start+ctg_length-k+i+Int32(1)];end
   mer_pos=temp_base;walk_length=Int32(0);result=UInt8(88)
   for step in Int32(0):maxwalk-Int32(1)
    bucket=Int32(murmur_key(temporary,mer_pos,k,maxwalk))
    while true
     slot=visited_base+bucket+Int32(1)
     @inbounds empty=visited_len[slot]==EMPTY_KEY
     if empty
      @inbounds visited_len[slot]=UInt32(k);visited_pos[slot]=mer_pos;visited_val[slot]=1
      break
     elseif (@inbounds visited_len[slot]==UInt32(k)&&equal_key(temporary,visited_pos[slot],temporary,mer_pos,k))
      result=UInt8(82);break
     end
     bucket=(bucket+Int32(1))%maxwalk
    end
    result==UInt8(82)&&break
    bucket=Int32(murmur_key(temporary,mer_pos,k,capacity));found=false;extension=UInt8(88)
    while true
     slot=table_base+bucket+Int32(1)
     @inbounds empty=keylen[slot]==EMPTY_KEY
     empty&&break
     @inbounds if keylen[slot]==UInt32(k)&&equal_key(reads,keypos[slot],temporary,mer_pos,k)
      found=true;extension=exts[slot];break
     end
     bucket=(bucket+Int32(1))%capacity
    end
    if !found;result=UInt8(88);break;end
    if extension==UInt8(70)||extension==UInt8(88);result=extension;break;end
    mer_pos+=Int32(1);@inbounds temporary[mer_pos+k]=extension
    extension!=0&&(walk_length+=Int32(1))
   end
   if walk_length>longest_length
    for i in Int32(0):walk_length-Int32(1);@inbounds longest[longest_base+i+Int32(1)]=temporary[temp_base+k+i+Int32(1)];end
    longest_length=walk_length
   end
   if result==UInt8(88)
    shift==Int32(8)&&break
    shift=Int32(-8)
   else
    shift==Int32(-8)&&break
    k>ctg_length&&break
    shift=Int32(8)
   end
  end
  CUDA.sync_warp(FULL_WARP)
  (CUDA.active_mask()&UInt32(1))==0&&break
  # CUDA.jl shfl_sync indexes lanes from1; Julia lane1 is native CUDA lane0.
  shift=CUDA.shfl_sync(FULL_WARP,shift,Int32(1));k+=shift
 end
 if lane==0;@inbounds walklens[group+Int32(1)]=UInt32(longest_length);end
 nothing
end
@inline function capture_hash_body!(i,keylen,keypos,reads,hashes,capacity,total)
 if i<total
  @inbounds hashes[i+Int32(1)]=keylen[i+Int32(1)]==EMPTY_KEY ? UInt32(0) : murmur_key(reads,keypos[i+Int32(1)],Int32(keylen[i+Int32(1)]),capacity)
 end
 nothing
end
