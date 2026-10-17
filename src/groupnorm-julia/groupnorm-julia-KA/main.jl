using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function forward_ka!(x,weight,bias,out,mean,rstd)
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1);gp=Int32(1024);img=Int32(256);gs=Int32(4);ng=Int32(8)
 offset=block*gp;wbase=mod(block,ng)*gs;scratch=@localmem Float32 (1024,2)
 first=0.f0;second=0.f0
 i=tid
 while i<gp
  value=@inbounds x[offset+i+Int32(1)];first+=value;second=muladd(value,value,second);i+=Int32(1024)
 end
 @inbounds scratch[tid+Int32(1),1]=first; scratch[tid+Int32(1),2]=second
 @synchronize
 for stride in (Int32(512),Int32(256),Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride
   @inbounds scratch[tid+Int32(1),1]+=scratch[tid+stride+Int32(1),1];scratch[tid+Int32(1),2]+=scratch[tid+stride+Int32(1),2]
  end
  @synchronize
 end
 first=@inbounds scratch[1,1];second=@inbounds scratch[1,2]
 @synchronize
 mu=first/Float32(gp);variance=muladd(-mu,mu,second/Float32(gp));rs=inv(sqrt(variance+1.f-5))
 if tid==Int32(0);@inbounds mean[block+Int32(1)]=mu;rstd[block+Int32(1)]=rs;end
 i=tid
 while i<gp
  channel=div(i,img);value=@inbounds x[offset+i+Int32(1)]
  @inbounds out[offset+i+Int32(1)]=muladd(rs*(value-mu),weight[wbase+channel+Int32(1)],bias[wbase+channel+Int32(1)])
  i+=Int32(1024)
 end
end
@kernel function forward2_ka!(x,weight,bias,out,mean,rstd)
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1);gp=Int32(1024);img=Int32(256);gs=Int32(4);ng=Int32(8)
 offset=block*gp;wbase=mod(block,ng)*gs;scratch=@localmem Float32 (1024,2)
 a0=0.f0;a1=0.f0;a2=0.f0;a3=0.f0;b0=0.f0;b1=0.f0;b2=0.f0;b3=0.f0;i=tid
 while i<div(gp,Int32(4))
  j=offset+Int32(4)*i
  v0=@inbounds x[j+Int32(1)];v1=@inbounds x[j+Int32(2)];v2=@inbounds x[j+Int32(3)];v3=@inbounds x[j+Int32(4)]
  a0+=v0;a1+=v1;a2+=v2;a3+=v3;b0=muladd(v0,v0,b0);b1=muladd(v1,v1,b1);b2=muladd(v2,v2,b2);b3=muladd(v3,v3,b3);i+=Int32(1024)
 end
 first=((a0+a1)+a2)+a3;second=((b0+b1)+b2)+b3
 @inbounds scratch[tid+Int32(1),1]=first; scratch[tid+Int32(1),2]=second
 @synchronize
 for stride in (Int32(512),Int32(256),Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride
   @inbounds scratch[tid+Int32(1),1]+=scratch[tid+stride+Int32(1),1];scratch[tid+Int32(1),2]+=scratch[tid+stride+Int32(1),2]
  end
  @synchronize
 end
 first=@inbounds scratch[1,1];second=@inbounds scratch[1,2]
 @synchronize
 mu=first/Float32(gp);variance=muladd(-mu,mu,second/Float32(gp));rs=inv(sqrt(variance+1.f-5))
 if tid==Int32(0);@inbounds mean[block+Int32(1)]=mu;rstd[block+Int32(1)]=rs;end
 i=tid
 while i<gp
  channel=div(i,img);value=@inbounds x[offset+i+Int32(1)]
  @inbounds out[offset+i+Int32(1)]=muladd(rs*(value-mu),weight[wbase+channel+Int32(1)],bias[wbase+channel+Int32(1)])
  i+=Int32(1024)
 end
end
@kernel function backward_ka!(dout,x,mean,rstd,weight,dx,dweight,dbias)
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1);gp=Int32(1024);img=Int32(256);gs=Int32(4);ng=Int32(8)
 offset=block*gp;wbase=mod(block,ng)*gs;scratch=@localmem Float32 (1024,2)
 mu=@inbounds mean[block+Int32(1)];rs=@inbounds rstd[block+Int32(1)];first=0.f0;second=0.f0;i=tid
 while i<gp
  channel=div(i,img);dy=@inbounds dout[offset+i+Int32(1)];value=@inbounds x[offset+i+Int32(1)];gamma=@inbounds weight[wbase+channel+Int32(1)]
  wd=gamma*dy;norm=(value-mu)*rs;first+=wd;second=muladd(wd,norm,second);i+=Int32(1024)
 end
 @inbounds scratch[tid+Int32(1),1]=first; scratch[tid+Int32(1),2]=second
 @synchronize
 for stride in (Int32(512),Int32(256),Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride
   @inbounds scratch[tid+Int32(1),1]+=scratch[tid+stride+Int32(1),1];scratch[tid+Int32(1),2]+=scratch[tid+stride+Int32(1),2]
  end
  @synchronize
 end
 first=@inbounds scratch[1,1];second=@inbounds scratch[1,2]
 @synchronize
 wd_mean=first/Float32(gp);wdn_mean=second/Float32(gp);i=tid
 while i<gp
  channel=div(i,img);dy=@inbounds dout[offset+i+Int32(1)];value=@inbounds x[offset+i+Int32(1)];gamma=@inbounds weight[wbase+channel+Int32(1)]
  norm=(value-mu)*rs;@inbounds dx[offset+i+Int32(1)]=(gamma*dy-wd_mean-norm*wdn_mean)*rs;i+=Int32(1024)
 end
 for channel in Int32(0):Int32(3)
  first=0.f0;second=0.f0;i=tid
  while i<img
   dy=@inbounds dout[offset+channel*img+i+Int32(1)];value=@inbounds x[offset+channel*img+i+Int32(1)]
   first=muladd(dy,(value-mu)*rs,first);second+=dy;i+=Int32(1024)
  end
 @inbounds scratch[tid+Int32(1),1]=first; scratch[tid+Int32(1),2]=second
 @synchronize
 for stride in (Int32(512),Int32(256),Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<stride
   @inbounds scratch[tid+Int32(1),1]+=scratch[tid+stride+Int32(1),1];scratch[tid+Int32(1),2]+=scratch[tid+stride+Int32(1),2]
  end
  @synchronize
 end
 first=@inbounds scratch[1,1];second=@inbounds scratch[1,2]
 @synchronize
  if tid==Int32(0)
   Atomix.@atomic dweight[wbase+channel+Int32(1)]+=first
   Atomix.@atomic dbias[wbase+channel+Int32(1)]+=second
  end
 end
end
function factory(x,weight,bias,out,mean,rstd,dout,dx,dweight,dbias,mode)
 if mode=="backward"
  launch=backward_ka!(get_backend(x),1024)
  return ()->launch(dout,x,mean,rstd,weight,dx,dweight,dbias;ndrange=32768)
 end
 kernel=mode=="forward2" ? forward2_ka! : forward_ka!;launch=kernel(get_backend(x),1024)
 return ()->launch(x,weight,bias,out,mean,rstd;ndrange=32768)
end
groupnorm_study(factory)
