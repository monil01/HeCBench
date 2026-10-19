# Volatile scalar pointer helpers adapted from the previously audited BH port.
# These custom LLVM operations are explicit device extensions in ALL three models.
@generated function gvload(p::Core.LLVMPtr{UInt32,A})where A
 ir="define i32 @entry(i64 %x) { %p = inttoptr i64 %x to ptr addrspace($A)\n %v = load volatile i32, ptr addrspace($A) %p, align 4\n ret i32 %v }"
 :(Base.llvmcall(($ir,"entry"),UInt32,Tuple{UInt},UInt(p)))
end
@generated function gvstore(p::Core.LLVMPtr{UInt32,A},x::UInt32)where A
 ir="define void @entry(i64 %x, i32 %v) { %p = inttoptr i64 %x to ptr addrspace($A)\n store volatile i32 %v, ptr addrspace($A) %p, align 4\n ret void }"
 :(Base.llvmcall(($ir,"entry"),Cvoid,Tuple{UInt,UInt32},UInt(p),x))
end
@inline gg(a,i)=gvload(pointer(a,i+Int32(1)))
@inline gs(a,i,x)=gvstore(pointer(a,i+Int32(1)),UInt32(x))
@inline warp_sync()=CUDA.sync_warp(UInt32(0xffffffff))
@inline lock_state(x)=x==0 ? UInt32(2) : x==typemax(UInt32) ? UInt32(1) : UInt32(3)
@inline function key_hash(sh,keypos,trial,::Val{IP})where IP
 h=UInt64(0)
 for j in Int32(0):Int32(2IP-1)
  word=gg(sh,keypos+j÷Int32(2));part=(word>>((j%Int32(2))*Int32(16)))&UInt32(0xffff)
  h=h*UInt64(453570)+UInt64(part)
 end
 h+UInt64(UInt32(5696063)*UInt32(trial)*UInt32(trial))
end
function add_body!(table,nentries,keys,nkeys,failed,numfailure,maxfailure,sh,block,lane,::Val{IP})where IP
 ipe=Int32(IP+1);entries=Int32(32÷(IP+1));sub=lane%ipe;entry=lane÷ipe
 matchstart=Int32(2);lockstart=Int32(2)+entries;keystart=Int32(2)+Int32(2)*entries
 i=lane
 while i<Int32(IP*32)&&block*Int32(IP*32)+i<nkeys*Int32(IP)
  gs(sh,keystart+i,keys[block*Int32(IP*32)+i+Int32(1)]);i+=Int32(32)
 end
 warp_sync()
 current=Int32(0)
 while current<Int32(32)&&block*Int32(32)+current<nkeys
  pos=keystart+current*Int32(IP)
  lane==0&&gs(sh,Int32(0),UInt32(0));warp_sync()
  trial=Int32(0)
  while trial<Int32(30)&&gg(sh,Int32(0))==0
   base=Int32(key_hash(sh,pos,trial,Val(IP))%UInt64(nentries));loaded=min(entries,nentries-base);active=lane<ipe*loaded;address=base*ipe+lane
   lane<entries&&gs(sh,matchstart+lane,UInt32(1));lane==0&&gs(sh,Int32(1),typemax(UInt32));warp_sync()
   if active
    x=gg(table,address)
    if sub<Int32(IP)
     x!=gg(sh,pos+sub)&&gs(sh,matchstart+entry,UInt32(0))
    else
     gs(sh,lockstart+entry,lock_state(x))
    end
   end
   warp_sync()
   if active&&sub==0&&gg(sh,matchstart+entry)!=0&&gg(sh,lockstart+entry)==UInt32(3)
    gs(sh,Int32(1),UInt32(entry))
   end
   warp_sync()
   found=gg(sh,Int32(1))
   if active&&found!=typemax(UInt32)&&UInt32(entry)==found&&sub==Int32(IP)
    CUDA.atomic_add!(pointer(table,address+Int32(1)),UInt32(1));gs(sh,Int32(0),UInt32(1))
   end
   warp_sync()
   probe=Int32(0)
   while probe<loaded&&gg(sh,Int32(0))==0
    while gg(sh,lockstart+probe)==UInt32(1)
     x=active ? gg(table,address) : UInt32(0)
     active&&sub==Int32(IP)&&gs(sh,matchstart+entry,UInt32(1));warp_sync()
     if active
      if sub<Int32(IP)
       x!=gg(sh,pos+sub)&&gs(sh,matchstart+entry,UInt32(0))
      else
       gs(sh,lockstart+entry,lock_state(x))
      end
     end
     warp_sync()
    end
    state=gg(sh,lockstart+probe);retry=false
    if state==UInt32(3)
     if gg(sh,matchstart+probe)!=0&&active&&entry==probe&&sub==Int32(IP)
      CUDA.atomic_add!(pointer(table,address+Int32(1)),UInt32(1));gs(sh,Int32(0),UInt32(1))
     end
     warp_sync()
    elseif state==UInt32(2)
     if active&&entry==probe&&sub==Int32(IP)
      old=CUDA.atomic_cas!(pointer(table,address+Int32(1)),UInt32(0),typemax(UInt32));gs(sh,lockstart+probe,old==0 ? UInt32(3) : UInt32(1))
     end
     warp_sync()
     if gg(sh,lockstart+probe)==UInt32(3)
      if active&&entry==probe
       sub<Int32(IP)&&gs(table,address,gg(sh,pos+sub));CUDA.threadfence()
      end
      warp_sync()
      if active&&entry==probe&&sub==Int32(IP)
       gs(table,address,UInt32(1));gs(sh,Int32(0),UInt32(1))
      end
      warp_sync()
     else
      retry=true
     end
    end
    !retry&&(probe+=Int32(1));warp_sync()
   end
   trial+=Int32(1);warp_sync()
  end
  if gg(sh,Int32(0))==0&&lane==0
   slot=CUDA.atomic_add!(pointer(numfailure,1),UInt64(1))
   if slot<UInt64(maxfailure)
    for j in Int32(0):Int32(IP-1);failed[Int(slot)*IP+Int(j)+1]=gg(sh,pos+j);end
   end
  end
  warp_sync();current+=Int32(1)
 end
 nothing
end
