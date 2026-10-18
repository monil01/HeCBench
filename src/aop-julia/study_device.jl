# NVIDIA Longstaff-Schwartz source operation/mapping restoration. See original notice.
include(joinpath(@__DIR__,"study_svd.jl"))
@inline payoff(S,K,put)=put ? max(K-S,0.0) : max(S-K,0.0)
@inline itm(S,K,put)=put ? S<K : S>K
@inline function paths_body!(samples,paths,n,steps,K,S0,r,sigma,dt,put,g)
 if g<n
  S=S0;offset=g+1
  a=(r-.5*sigma*sigma)*dt;b=sigma*sqrt(dt)
  for step in 1:steps-1
   S=S*exp(a+b*samples[offset]);paths[offset]=S;offset+=n
  end
  S=S*exp(a+b*samples[offset]);paths[offset]=payoff(S,K,put)
 end
 nothing
end
@inline function prepare_body!(paths,flags,svds,n,K,put,timestep,tid,ints,mem,sync)
 # Integer scratch: scan_input256, scan_output257, atomic reduction1.
 # Float scratch: 12 source R/W slots, four atomic moments.
 if tid<12;mem[tid+1]=0.;end
 sync()
 found=Int32(0);m=Int32(0);s1=0.;s2=0.;s3=0.;s4=0.
 path=tid
 while path<n
  S=paths[timestep*n+path+1];inside=itm(S,K,put);ints[tid+1]=Int32(inside)
  sync()
  if tid==0
   ints[257]=0
   for j in 1:256;ints[257+j]=ints[256+j]+ints[j];end
  end
  sync()
  partial=ints[257+tid];total=ints[513]
  if found<3
   if inside && found+partial<3;mem[found+partial+1]=S;end
   sync();found+=total
  end
  if tid==0;ints[514]=0;end
  sync();Atomix.@atomic ints[514] |= Int32(inside);sync()
  if ints[514]!=0
   m+=Int32(inside);x=inside ? S : 0.;xx=x*x
   s1+=x;s2+=xx;s3+=xx*x;s4+=xx*xx
  end
  path+=Int32(256)
 end
 sync()
 if tid==0;ints[514]=0;end
 sync();Atomix.@atomic ints[514] += m;sync()
 # Retain the actual thread-local source flag and branch, including its partial writes on skip.
 notenough=tid==0 && ints[514]<4
 if notenough
  flags[timestep+1]=1
 else
  if tid==0
   for j in 13:16;mem[j]=0.;end
  end
  sync()
  Atomix.@atomic mem[13] += s1
  Atomix.@atomic mem[14] += s2
  Atomix.@atomic mem[15] += s3
  Atomix.@atomic mem[16] += s4
  sync()
  if tid==0;svd_3x3(ints[514],(mem[13],mem[14],mem[15],mem[16]),mem);end
  sync()
  if tid<12;svds[16*timestep+tid+1]=mem[tid+1];end
 end
 nothing
end
@inline function partial_beta_body!(paths,flags,svds,temp,n,timestep,K,put,block,tid,mem,sync)
 if flags[timestep+1]!=0;return nothing;end
 if tid<12;mem[tid+1]=svds[16*timestep+tid+1];end
 sync()
 R00=mem[1];R01=mem[2];R02=mem[3];R11=mem[4];R12=mem[5];R22=mem[6]
 W00=mem[7];W01=mem[8];W02=mem[9];W11=mem[10];W12=mem[11];W22=mem[12]
 i00=R00!=0.0 ? 1.0/R00 : 0.;i11=R11!=0.0 ? 1.0/R11 : 0.;i22=R22!=0.0 ? 1.0/R22 : 0.
 i01=i00*i11*R01;i02=i00*i22*R02;i12=i22*R12;w00=W00*i00
 b0=0.;b1=0.;b2=0.;path=block*Int32(128)+tid
 while path<n
  S=paths[timestep*n+path+1];inside=itm(S,K,put)
  q1=i11*S-i01;q2=i22*S*S-i02-q1*i12
  w0=w00+W01*q1+W02*q2;w1=W11*q1+W12*q2;w2=W22*q2
  cash=inside ? paths[end-n+path+1] : 0.
  b0+=w0*cash;b1+=w1*cash;b2+=w2*cash
  path+=Int32(128*128)
 end
 if tid==0;mem[13]=0.;mem[14]=0.;mem[15]=0.;end
 sync()
 Atomix.@atomic mem[13] += b0
 Atomix.@atomic mem[14] += b1
 Atomix.@atomic mem[15] += b2
 sync()
 if tid==0
  temp[block+1]=mem[13];temp[128+block+1]=mem[14];temp[256+block+1]=mem[15]
 end
 nothing
end
@inline function final_beta_body!(flags,temp,timestep,tid,mem,sync)
 if flags[timestep+1]!=0
  if tid<3;temp[tid+1]=0.;end
  return nothing
 end
 b0=temp[tid+1];b1=temp[128+tid+1];b2=temp[256+tid+1]
 if tid==0;mem[1]=0.;mem[2]=0.;mem[3]=0.;end
 sync()
 Atomix.@atomic mem[1] += b0
 Atomix.@atomic mem[2] += b1
 Atomix.@atomic mem[3] += b2
 sync()
 if tid==0;temp[1]=mem[1];temp[2]=mem[2];temp[3]=mem[3];end
 nothing
end
@inline function cash_body!(paths,flags,temp,n,timestep,K,put,discount,g,stride)
 skip=flags[timestep+1]!=0;b0=temp[1];b1=temp[2];b2=temp[3];path=g
 while path<n
  old=discount*paths[end-n+path+1]
  if skip;paths[end-n+path+1]=old
  else
   S=paths[timestep*n+path+1];p=payoff(S,K,put);estimated=(b0+b1*S+b2*S*S)*discount
   paths[end-n+path+1]=(p<=1.e-8 || p<=estimated) ? old : p
  end
  path+=stride
 end
 nothing
end
@inline function partial_sum_body!(paths,temp,n,block,tid,mem,sync)
 path=block*Int32(128)+tid;s=path<n ? paths[end-n+path+1] : 0.
 if tid==0;mem[1]=0.;end
 sync();Atomix.@atomic mem[1] += s;sync()
 if tid==0;temp[block+1]=mem[1];end
 nothing
end
@inline function final_sum_body!(temp,n,blocks,discount,tid,mem,sync)
 s=0.;item=tid
 while item<blocks;s+=temp[item+1];item+=Int32(128);end
 if tid==0;mem[1]=0.;end
 sync();Atomix.@atomic mem[1] += s;sync()
 if tid==0;temp[1]=discount*mem[1]/Float64(n);end
 nothing
end
