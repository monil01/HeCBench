@inline function cycle_body!(tid,A,B,b,input,finished,head,s)
 m=A*B-Int32(1)
 if tid==0;@inbounds s[2]=CUDA.atomic_add!(pointer(head),Int32(1));end
 block_barrier()
 while s[2]<m
  @inbounds next=s[2]*A-m*(s[2]÷B)
  if next==s[2]
   block_barrier()
   if tid==0;@inbounds s[2]=CUDA.atomic_add!(pointer(head),Int32(1));end
   block_barrier();continue
  end
  data1=0f0;data2=0f0;data3=0f0;data4=0f0;i=tid
  if i<b;@inbounds data1=input[s[2]*b+i+1];end;i+=Int32(64)
  if i<b;@inbounds data2=input[s[2]*b+i+1];end;i+=Int32(64)
  if i<b;@inbounds data3=input[s[2]*b+i+1];end;i+=Int32(64)
  if i<b;@inbounds data4=input[s[2]*b+i+1];end
  if tid==0;@inbounds s[1]=CUDA.atomic_add!(pointer(finished,Int(s[2])+1),Int32(0));end
  block_barrier()
  while s[1]==0
   backup1=0f0;backup2=0f0;backup3=0f0;backup4=0f0;i=tid
   if i<b;@inbounds backup1=input[next*b+i+1];end;i+=Int32(64)
   if i<b;@inbounds backup2=input[next*b+i+1];end;i+=Int32(64)
   if i<b;@inbounds backup3=input[next*b+i+1];end;i+=Int32(64)
   if i<b;@inbounds backup4=input[next*b+i+1];end
   block_barrier()
   if tid==0;@inbounds s[1]=CUDA.atomic_xchg!(pointer(finished,Int(next)+1),Int32(1));end
   block_barrier()
   if s[1]==0
    i=tid;if i<b;@inbounds input[next*b+i+1]=data1;end;i+=Int32(64)
    if i<b;@inbounds input[next*b+i+1]=data2;end;i+=Int32(64)
    if i<b;@inbounds input[next*b+i+1]=data3;end;i+=Int32(64)
    if i<b;@inbounds input[next*b+i+1]=data4;end
   end
   data1=backup1;data2=backup2;data3=backup3;data4=backup4;next=next*A-m*(next÷B)
  end
  if tid==0;@inbounds s[2]=CUDA.atomic_add!(pointer(head),Int32(1));end
  block_barrier()
 end
 return
end
