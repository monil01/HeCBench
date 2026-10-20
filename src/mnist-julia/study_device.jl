# Source-aligned source19 CNN GPU bodies; native notices preserved in original_native/.
@inline function body_apply_step_function!(pos,size,input,output,N)
 Nsource=N
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  output[n+Int32(1)]=1f0/(1f0+exp(-input[n+Int32(1)]))
 end
 nothing
end
@inline function body_makeError!(pos,size,err,output,Y,N)
 Nsource=N
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  err[n+Int32(1)]=(Y==n ? 1f0 : 0f0)-output[n+Int32(1)]
 end
 nothing
end
@inline function body_apply_grad!(pos,size,output,grad,N)
 Nsource=N
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  output[n+Int32(1)]=muladd(.1f0,grad[n+Int32(1)],output[n+Int32(1)])
 end
 nothing
end
@inline function body_fp_preact_c1!(pos,size,input,preact,weight)
 Nsource=Int32(86400)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  kr=n%Int32(5);kc=div(n,Int32(5))%Int32(5);ch=div(n,Int32(25))%Int32(6);r=div(n,Int32(150))%Int32(24);c=div(n,Int32(3600))%Int32(24)
  atomic_add!(preact,(ch*Int32(24)+r)*Int32(24)+c+Int32(1),weight[(ch*Int32(5)+kr)*Int32(5)+kc+Int32(1)]*input[(r+kr)*Int32(28)+c+kc+Int32(1)])
 end
 nothing
end
@inline function body_fp_bias_c1!(pos,size,preact,bias)
 Nsource=Int32(3456)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);r=div(n,Int32(6))%Int32(24);c=div(n,Int32(144))%Int32(24);i=(ch*Int32(24)+r)*Int32(24)+c+Int32(1)
  preact[i]+=bias[ch+Int32(1)]
 end
 nothing
end
@inline function body_fp_preact_s1!(pos,size,input,preact,weight)
 Nsource=Int32(3456)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  kr=n%Int32(4);kc=div(n,Int32(4))%Int32(4);ch=div(n,Int32(16))%Int32(6);r=div(n,Int32(96))%Int32(6);c=div(n,Int32(576))%Int32(6)
  atomic_add!(preact,(ch*Int32(6)+r)*Int32(6)+c+Int32(1),weight[kr*Int32(4)+kc+Int32(1)]*input[(ch*Int32(24)+r*Int32(4)+kr)*Int32(24)+c*Int32(4)+kc+Int32(1)])
 end
 nothing
end
@inline function body_fp_bias_s1!(pos,size,preact,bias)
 Nsource=Int32(216)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);r=div(n,Int32(6))%Int32(6);c=div(n,Int32(36))%Int32(6);i=(ch*Int32(6)+r)*Int32(6)+c+Int32(1)
  preact[i]+=bias[Int32(1)]
 end
 nothing
end
@inline function body_fp_preact_f!(pos,size,input,preact,weight)
 Nsource=Int32(2160)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  cl=n%Int32(10);ch=div(n,Int32(10))%Int32(6);r=div(n,Int32(60))%Int32(6);c=div(n,Int32(360))%Int32(6);i=(ch*Int32(6)+r)*Int32(6)+c
  atomic_add!(preact,cl+Int32(1),weight[cl*Int32(216)+i+Int32(1)]*input[i+Int32(1)])
 end
 nothing
end
@inline function body_fp_bias_f!(pos,size,preact,bias)
 Nsource=Int32(10)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  preact[n+Int32(1)]+=bias[n+Int32(1)]
 end
 nothing
end
@inline function body_bp_weight_f!(pos,size,dweight,dpreact,poutput)
 Nsource=Int32(2160)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  cl=n%Int32(10);ch=div(n,Int32(10))%Int32(6);r=div(n,Int32(60))%Int32(6);c=div(n,Int32(360))%Int32(6);i=(ch*Int32(6)+r)*Int32(6)+c
  dweight[cl*Int32(216)+i+Int32(1)]=dpreact[cl+Int32(1)]*poutput[i+Int32(1)]
 end
 nothing
end
@inline function body_bp_bias_f!(pos,size,bias,dpreact)
 Nsource=Int32(10)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  bias[n+Int32(1)]=muladd(.1f0,dpreact[n+Int32(1)],bias[n+Int32(1)])
 end
 nothing
end
@inline function body_bp_output_s1!(pos,size,doutput,nweight,ndpreact)
 Nsource=Int32(2160)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  cl=n%Int32(10);ch=div(n,Int32(10))%Int32(6);r=div(n,Int32(60))%Int32(6);c=div(n,Int32(360))%Int32(6);i=(ch*Int32(6)+r)*Int32(6)+c
  atomic_add!(doutput,i+Int32(1),nweight[cl*Int32(216)+i+Int32(1)]*ndpreact[cl+Int32(1)])
 end
 nothing
end
@inline function body_bp_preact_s1!(pos,size,dpreact,doutput,preact)
 Nsource=Int32(216)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);r=div(n,Int32(6))%Int32(6);c=div(n,Int32(36))%Int32(6);i=(ch*Int32(6)+r)*Int32(6)+c+Int32(1);o=1f0/(1f0+exp(-preact[i]))
  dpreact[i]=doutput[i]*o*(1f0-o)
 end
 nothing
end
@inline function body_bp_weight_s1!(pos,size,dweight,dpreact,poutput)
 Nsource=Int32(3456)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  kr=n%Int32(4);kc=div(n,Int32(4))%Int32(4);ch=div(n,Int32(16))%Int32(6);r=div(n,Int32(96))%Int32(6);c=div(n,Int32(576))%Int32(6)
  atomic_add!(dweight,kr*Int32(4)+kc+Int32(1),dpreact[(ch*Int32(6)+r)*Int32(6)+c+Int32(1)]*poutput[(ch*Int32(24)+r*Int32(4)+kr)*Int32(24)+c*Int32(4)+kc+Int32(1)])
 end
 nothing
end
@inline function body_bp_bias_s1!(pos,size,bias,dpreact)
 Nsource=Int32(216)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);r=div(n,Int32(6))%Int32(6);c=div(n,Int32(36))%Int32(6)
  atomic_add!(bias,Int32(1),.1f0*dpreact[(ch*Int32(6)+r)*Int32(6)+c+Int32(1)]/216f0)
 end
 nothing
end
@inline function body_bp_output_c1!(pos,size,doutput,nweight,ndpreact)
 Nsource=Int32(3456)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  kr=n%Int32(4);kc=div(n,Int32(4))%Int32(4);ch=div(n,Int32(16))%Int32(6);r=div(n,Int32(96))%Int32(6);c=div(n,Int32(576))%Int32(6)
  atomic_add!(doutput,(ch*Int32(24)+r*Int32(4)+kr)*Int32(24)+c*Int32(4)+kc+Int32(1),nweight[kr*Int32(4)+kc+Int32(1)]*ndpreact[(ch*Int32(6)+r)*Int32(6)+c+Int32(1)])
 end
 nothing
end
@inline function body_bp_preact_c1!(pos,size,dpreact,doutput,preact)
 Nsource=Int32(3456)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);r=div(n,Int32(6))%Int32(24);c=div(n,Int32(144))%Int32(24);i=(ch*Int32(24)+r)*Int32(24)+c+Int32(1);o=1f0/(1f0+exp(-preact[i]))
  dpreact[i]=doutput[i]*o*(1f0-o)
 end
 nothing
end
@inline function body_bp_weight_c1!(pos,size,dweight,dpreact,poutput)
 Nsource=Int32(86400)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);kr=div(n,Int32(6))%Int32(5);kc=div(n,Int32(30))%Int32(5);r=div(n,Int32(150))%Int32(24);c=div(n,Int32(3600))%Int32(24)
  atomic_add!(dweight,(ch*Int32(5)+kr)*Int32(5)+kc+Int32(1),dpreact[(ch*Int32(24)+r)*Int32(24)+c+Int32(1)]*poutput[(r+kr)*Int32(28)+c+kc+Int32(1)]/576f0)
 end
 nothing
end
@inline function body_bp_bias_c1!(pos,size,bias,dpreact)
 Nsource=Int32(3456)
 @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)
  ch=n%Int32(6);r=div(n,Int32(6))%Int32(24);c=div(n,Int32(144))%Int32(24)
  atomic_add!(bias,ch+Int32(1),.1f0*dpreact[(ch*Int32(24)+r)*Int32(24)+c+Int32(1)]/576f0)
 end
 nothing
end
