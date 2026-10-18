# Source final_gpu_kernel: two matrices per lane, shared Hamiltonian and ping-pair sigma.
@inline function final_body!(sigma_in,sigma_out,ham,num,BX,TX,TY,sh,barrier)
 ij=TX;i=ij÷Int32(7);j=ij%Int32(7);sub=TY;start=BX*Int32(32)+sub*Int32(16);stop=min(num,start+Int32(16))
 if ij<Int32(49)&&sub==0
  @inbounds h=ham[ij+1];sh[ij+1]=h.data[1].value;sh[Int32(49)+ij+1]=h.data[2].value
 end
 for m in start:Int32(2):stop-Int32(1)
  barrier()
  a=StudyV2(0f0);b=StudyV2(0f0)
  if ij<Int32(49)
   @inbounds s1=sigma_in[m*Int32(49)+ij+1];s2=sigma_in[(m+Int32(1))*Int32(49)+ij+1]
   sh[Int32(98)+sub*Int32(49)+ij+1]=s1.data[1].value;sh[Int32(294)+sub*Int32(49)+ij+1]=s1.data[2].value
   sh[Int32(98)+(Int32(2)+sub)*Int32(49)+ij+1]=s2.data[1].value;sh[Int32(294)+(Int32(2)+sub)*Int32(49)+ij+1]=s2.data[2].value
   @inbounds old1=sigma_out[m*Int32(49)+ij+1];old2=sigma_out[(m+Int32(1))*Int32(49)+ij+1]
   a=StudyV2(old1.data[1].value,old2.data[1].value);b=StudyV2(old1.data[2].value,old2.data[2].value)
  end
  barrier()
  if ij<Int32(49)
   @inbounds for k in Int32(0):Int32(6)
    ik=i*Int32(7)+k;kj=k*Int32(7)+j
    s1=StudyV2(sh[Int32(98)+sub*Int32(49)+kj+1],sh[Int32(98)+(Int32(2)+sub)*Int32(49)+kj+1]);s2=StudyV2(sh[Int32(294)+sub*Int32(49)+kj+1],sh[Int32(294)+(Int32(2)+sub)*Int32(49)+kj+1]);a+=sh[ik+1]*s2;a+=sh[Int32(49)+ik+1]*s1;b-=sh[ik+1]*s1;b+=sh[Int32(49)+ik+1]*s2
    s1=StudyV2(sh[Int32(98)+sub*Int32(49)+ik+1],sh[Int32(98)+(Int32(2)+sub)*Int32(49)+ik+1]);s2=StudyV2(sh[Int32(294)+sub*Int32(49)+ik+1],sh[Int32(294)+(Int32(2)+sub)*Int32(49)+ik+1]);a-=sh[kj+1]*s2;a+=sh[Int32(49)+kj+1]*s1;b+=sh[kj+1]*s1;b-=sh[Int32(49)+kj+1]*s2
   end
   @inbounds sigma_out[m*Int32(49)+ij+1]=StudyV2(a.data[1].value,b.data[1].value);sigma_out[(m+Int32(1))*Int32(49)+ij+1]=StudyV2(a.data[2].value,b.data[2].value)
  end
 end
 return
end
