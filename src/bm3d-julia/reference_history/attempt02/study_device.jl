using StaticArrays
@inline function match_lane!(g,img,stacks,counts)
 gx=(g-1)%64;gy=(g-1)÷64;px=gx*3;py=gy*3
 if 105<=px<107;px=104;end
 if 33<=py<35;py=32;end
 if px<105&&py<33
  best=MVector{15,UInt32}(ntuple(_->typemax(UInt32),15));found=0;base=(gx÷32)*96
  for dy in max(-19,-py):min(19,32-py),dx in max(-19,-base):min(19,104-px)
   if !(dx==0&&dy==0)
    diff=UInt32(0)
    for y in 0:7,x in 0:7
     a=Int32(@inbounds img[px+x+(py+y)*112+1]);b=Int32(@inbounds img[px+dx+x+(py+dy+y)*112+1]);d=a-b;diff+=UInt32(d*d)
    end
    if diff<UInt32(160000)
     key=((diff>>2)<<16)|(UInt32(dy&255)<<8)|UInt32(dx&255)
     if key<best[15]
      j=15
      while j>1&&key<best[j-1];best[j]=best[j-1];j-=1;end
      best[j]=key
     end
     found=min(found+1,15)
    end
   end
  end
  for z in 1:found;@inbounds stacks[(g-1)*15+z]=UInt16(best[z]&UInt32(65535));end
  depth=1;while depth*2<=found+1;depth*=2;end
  @inbounds counts[g]=UInt32(depth-1)
 end
 nothing
end
@inline function origin(g)
 x=((g-1)%64)*3;y=((g-1)÷64)*3
 if 105<=x<107;x=104;end
 if 33<=y<35;y=32;end
 x,y
end
@inline function offset(stacks,g,z)
 if z==0;return (0,0);end
 v=@inbounds stacks[(g-1)*15+z];x=Int(v&UInt16(255));y=Int(v>>8);(x>=128 ? x-256 : x,y>=128 ? y-256 : y)
end
@inline function gather_lane!(t,img,stacks,counts,group)
 g=(t-1)÷64+1;l=(t-1)%64;x=l%8;y=l÷8;px,py=origin(g)
 if px<105&&py<33
  n=Int(@inbounds counts[g])+1
  for z in 0:n-1
   dx,dy=offset(stacks,g,z);@inbounds group[(g-1)*1024+z*64+l+1]=Float32(img[px+dx+x+(py+dy+y)*112+1])
  end
 end
 nothing
end

@inline function dctvec(v)
 X07P = v[1] + v[8]
 X16P = v[2] + v[7]
 X25P = v[3] + v[6]
 X34P = v[4] + v[5]
 X07M = v[1] - v[8]
 X61M = v[7] - v[2]
 X25M = v[3] - v[6]
 X43M = v[5] - v[4]
 X07P34PP = X07P + X34P
 X07P34PM = X07P - X34P
 X16P25PP = X16P + X25P
 X16P25PM = X16P - X25P
 (0.3535533905932737f0 * (X07P34PP + X16P25PP),
 0.3535533905932737f0 * (1.387039845322148f0 * X07M - 1.175875602419359f0 * X61M + 0.785694958387102f0 * X25M - 0.275899379282943f0 * X43M),
 0.3535533905932737f0 * (1.306562964876377f0 * X07P34PM + 0.541196100146197f0 * X16P25PM),
 0.3535533905932737f0 * (1.175875602419359f0 * X07M + 0.275899379282943f0 * X61M - 1.387039845322148f0 * X25M + 0.785694958387102f0 * X43M),
 0.3535533905932737f0 * (X07P34PP - X16P25PP),
 0.3535533905932737f0 * (0.785694958387102f0 * X07M + 1.387039845322148f0 * X61M + 0.275899379282943f0 * X25M - 1.175875602419359f0 * X43M),
 0.3535533905932737f0 * (0.541196100146197f0 * X07P34PM - 1.306562964876377f0 * X16P25PM),
 0.3535533905932737f0 * (0.275899379282943f0 * X07M + 0.785694958387102f0 * X61M + 1.175875602419359f0 * X25M + 1.387039845322148f0 * X43M))
end

@inline function idctvec(v)
 Y04P   = v[1] + v[5]
 Y2b6eP = 1.306562964876377f0 * v[3] + 0.541196100146197f0 * v[7]
 Y04P2b6ePP = Y04P + Y2b6eP
 Y04P2b6ePM = Y04P - Y2b6eP
 Y7f1aP3c5dPP = 0.275899379282943f0 * v[8] + 1.387039845322148f0 * v[2] + 1.175875602419359f0 * v[4] + 0.785694958387102f0 * v[6]
 Y7a1fM3d5cMP = 1.387039845322148f0 * v[8] - 0.275899379282943f0 * v[2] + 0.785694958387102f0 * v[4] - 1.175875602419359f0 * v[6]
 Y04M   = v[1] - v[5]
 Y2e6bM = 0.541196100146197f0 * v[3] - 1.306562964876377f0 * v[7]
 Y04M2e6bMP = Y04M + Y2e6bM
 Y04M2e6bMM = Y04M - Y2e6bM
 Y1c7dM3f5aPM = 1.175875602419359f0 * v[2] - 0.785694958387102f0 * v[8] - 0.275899379282943f0 * v[4] - 1.387039845322148f0 * v[6]
 Y1d7cP3a5fMM = 0.785694958387102f0 * v[2] + 1.175875602419359f0 * v[8] - 1.387039845322148f0 * v[4] + 0.275899379282943f0 * v[6]
 (0.3535533905932737f0 * (Y04P2b6ePP + Y7f1aP3c5dPP),
 0.3535533905932737f0 * (Y04M2e6bMP + Y1c7dM3f5aPM),
 0.3535533905932737f0 * (Y04M2e6bMM + Y1d7cP3a5fMM),
 0.3535533905932737f0 * (Y04P2b6ePM - Y7a1fM3d5cMP),
 0.3535533905932737f0 * (Y04P2b6ePM + Y7a1fM3d5cMP),
 0.3535533905932737f0 * (Y04M2e6bMM - Y1d7cP3a5fMM),
 0.3535533905932737f0 * (Y04M2e6bMP - Y1c7dM3f5aPM),
 0.3535533905932737f0 * (Y04P2b6ePP - Y7f1aP3c5dPP))
end

@inline function transform_lane!(tile,group,inverse)
 base=(tile-1)*64
 for y in 0:7
  v=ntuple(j->(@inbounds group[base+y*8+j]),8);w=inverse ? idctvec(v) : dctvec(v)
  for x in 1:8;@inbounds group[base+y*8+x]=w[x];end
 end
 for x in 0:7
  v=ntuple(j->(@inbounds group[base+(j-1)*8+x+1]),8);w=inverse ? idctvec(v) : dctvec(v)
  for y in 1:8;@inbounds group[base+(y-1)*8+x+1]=w[y];end
 end
 nothing
end
@inline function hard_lane!(g,group,counts,weights)
 px,py=origin(g)
 if px<105&&py<33
  n=Int(@inbounds counts[g])+1;nonzero=0;threshold=2.7f0*sqrt(Float32(n*625));v=MVector{16,Float32}(ntuple(_->0f0,16))
  for l in 0:63
   for z in 1:n;v[z]=@inbounds group[(g-1)*1024+(z-1)*64+l+1];end
   step=1
   while step<n
    for j in 1:2*step:n,k in 0:step-1;a=v[j+k];b=v[j+k+step];v[j+k]=a+b;v[j+k+step]=a-b;end
    step*=2
   end
   for z in 1:n
    if abs(v[z])<threshold;v[z]=0f0;else;nonzero+=1;end
   end
   step=1
   while step<n
    for j in 1:2*step:n,k in 0:step-1;a=v[j+k];b=v[j+k+step];v[j+k]=a+b;v[j+k+step]=a-b;end
    step*=2
   end
   for z in 1:n;@inbounds group[(g-1)*1024+(z-1)*64+l+1]=v[z]/Float32(n);end
  end
  @inbounds weights[g]=1f0/Float32(max(nonzero,1))
 end
 nothing
end
@inline function aggregate_lane!(t,group,stacks,counts,weights,win,num,den)
 g=(t-1)÷64+1;l=(t-1)%64;px,py=origin(g)
 if px<105&&py<33
  n=Int(@inbounds counts[g])+1;wp=@inbounds weights[g];window=@inbounds win[l+1]
  for z in 0:n-1
   dx,dy=offset(stacks,g,z);i=px+dx+l%8+(py+dy+l÷8)*112+1;v=@inbounds group[(g-1)*1024+z*64+l+1]
   model_atomic!(num,i,(v*window)*wp);model_atomic!(den,i,window*wp)
  end
 end
 nothing
end
@inline function final_lane!(i,num,den,out)
 @inbounds out[i]=UInt8(clamp(round(Int32,num[i]/den[i]),Int32(0),Int32(255)));nothing
end
