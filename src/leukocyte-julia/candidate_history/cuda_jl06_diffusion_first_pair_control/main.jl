using CUDA
const MODEL="cuda_jl"
include(joinpath(@__DIR__,"..","study_host.jl"))
function gicov_kernel!(gx,gy,si,co,tx,ty,out,m,n)
 gid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 localrows=m-44;groups=n-44
 if gid<localrows*groups
  x=gid÷localrows+22;y=gid%localrows+22;best=0.0f0
  for k=0:6
   total=0.0f0;mean=0.0f0;m2=0.0f0
   for p=0:149
    xx=x+tx[k*150+p+1];yy=y+ty[k*150+p+1];idx=xx*m+yy+1
    value=muladd(gx[idx],co[p+1],gy[idx]*si[p+1]);total+=value
    delta=value-mean;mean+=delta/Float32(p+1);m2=muladd(delta,value-mean,m2)
   end
   mean=total/150.0f0;variance=m2/149.0f0;score=(mean*mean)/variance
   if score>best;best=score;end
  end
  out[x*m+y+1]=best
 end;nothing
end
function dilation_kernel!(st,img,out,m,n)
 tid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1;r=tid%m;c=tid÷m
 if c<n
  maximum=0.0f0
  for sr=0:24
   y=r-12+sr
   if 0<=y<m
    for sc=0:24
     x=c-12+sc
     if 0<=x<n&&st[sr*25+sc+1]!=0.0f0
      v=img[x*m+y+1];if v>maximum;maximum=v;end
     end
    end
   end
  end
  out[r*n+c+1]=maximum
 end;nothing
end
function mgvf_kernel!(out,input,offsets,ms,ns,counts)
 tid=threadIdx().x-1;cell=blockIdx().x;off=offsets[cell];m=ms[cell];n=ns[cell];N=m*n;tiles=cld(N,Int32(256))
 field=CuStaticSharedArray(Float32,3321);buffer=CuStaticSharedArray(Float32,256);converged=CuStaticSharedArray(Int32,1)
 for tile=0:tiles-1
  p=tile*256+tid;if p<N;field[p+1]=out[off+p+1];end
 end
 sync_threads();if tid==0;converged[1]=0;end;sync_threads();iteration=Int32(0)
 while converged[1]==0&&iteration<500
  total=0.0f0
  for tile=0:tiles-1
   p=tile*256+tid;i=p÷n;j=p%n;old=0.0f0;new=0.0f0
   if i<m
    ru=max(i-1,Int32(0));rd=min(i+1,m-1);cl=max(j-1,Int32(0));cr=min(j+1,n-1)
    old=field[p+1];u=field[ru*n+j+1]-old;d=field[rd*n+j+1]-old;l=field[i*n+cl+1]-old;r=field[i*n+cr+1]-old
    ur=field[ru*n+cr+1]-old;dr=field[rd*n+cr+1]-old;ul=field[ru*n+cl+1]-old;dl=field[rd*n+cl+1]-old
    uh=muladd(atan((-u)*1.0f10),Float32(1.0f0/3.14159f0),0.5f0);dh=muladd(atan(d*1.0f10),Float32(1.0f0/3.14159f0),0.5f0)
    lh=muladd(atan((-l)*1.0f10),Float32(1.0f0/3.14159f0),0.5f0);rh=muladd(atan(r*1.0f10),Float32(1.0f0/3.14159f0),0.5f0)
    urh=muladd(atan((ur*0.0f0)*1.0f10),Float32(1.0f0/3.14159f0),0.5f0);drh=muladd(atan((dr*2.0f0)*1.0f10),Float32(1.0f0/3.14159f0),0.5f0)
    ulh=muladd(atan((ul*(-2.0f0))*1.0f10),Float32(1.0f0/3.14159f0),0.5f0);dlh=muladd(atan((dl*0.0f0)*1.0f10),Float32(1.0f0/3.14159f0),0.5f0)
    acc=muladd(uh,u,dh*d);acc=muladd(lh,l,acc);acc=muladd(rh,r,acc);acc=muladd(urh,ur,acc);acc=muladd(drh,dr,acc);acc=muladd(ulh,ul,acc);acc=muladd(dlh,dl,acc)
    new=muladd(0.1f0,acc,old);v=input[off+p+1]
    new=Float32(muladd((-0.2)*Float64(v),Float64(new-v),Float64(new)))
   end
   sync_threads()
   if tile>0&&p-256<N;field[p-256+1]=buffer[tid+1];end
   if tile<tiles-1;buffer[tid+1]=new;elseif i<m;field[p+1]=new;end
   total+=abs(new-old);sync_threads()
  end
  buffer[tid+1]=total;sync_threads()
  # Source NEXT_LOWEST_POWER_OF_TWO=256; the non-power branch is inactive.
  for stride=(128,64,32,16,8,4,2,1)
   if tid<stride;buffer[tid+1]+=buffer[tid+stride+1];end;sync_threads()
  end
  if tid==0&&buffer[1]/Float32(N)<1.0f-5;converged[1]=1;end
  sync_threads();iteration+=1
 end
 for tile=0:tiles-1
  p=tile*256+tid;if p<N;out[off+p+1]=field[p+1];end
 end
 if tid==0;counts[cell]=iteration;end;nothing
end
launch_gicov!(gx,gy,si,co,tx,ty,out,m,n)=(@cuda threads=256 blocks=cld((m-44)*(n-44),256) gicov_kernel!(gx,gy,si,co,tx,ty,out,Int32(m),Int32(n)))
launch_dilation!(st,img,out,m,n)=(@cuda threads=176 blocks=cld(m*n,176) dilation_kernel!(st,img,out,Int32(m),Int32(n)))
launch_mgvf!(out,input,offsets,ms,ns,counts,cells)=(@cuda threads=256 blocks=cells mgvf_kernel!(out,input,offsets,ms,ns,counts))
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
