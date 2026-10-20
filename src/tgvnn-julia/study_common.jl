using Printf
const TVLIB=joinpath(@__DIR__,"libtgv_vendor.so")
raw(a)=Ptr{Cvoid}(UInt(pointer(a)))
function device_array(::Type{T},n::Int) where T
 p=ccall((:tgv_alloc,TVLIB),Ptr{Cvoid},(Csize_t,),n*sizeof(T))
 p==C_NULL && error("device allocation")
 unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(n,);own=false)
end
free_array(a)=ccall((:tgv_free,TVLIB),Cvoid,(Ptr{Cvoid},),raw(a))
function h2d!(a,h)
 GC.@preserve h ccall((:tgv_h2d,TVLIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Csize_t),raw(a),pointer(h),sizeof(h))
end
copy_device!(a,b,n)=ccall((:tgv_d2d,TVLIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Csize_t),raw(a),raw(b),n*sizeof(eltype(a)))
zero_device!(a)=ccall((:tgv_zero,TVLIB),Cvoid,(Ptr{Cvoid},Csize_t),raw(a),length(a)*sizeof(eltype(a)))
function read_ra(path)
 open(path,"r") do io
  h=read!(io,Vector{UInt64}(undef,6));h[1]==0x7961727261776172 && h[2]==0 && h[3]==4 && h[4]==8 && h[6]==3 || error("RA header")
  dims=Tuple(Int.(read!(io,Vector{UInt64}(undef,3))));n=prod(dims)
  h[5]==8n && filesize(path)==72+8n || error("RA extent")
  a=read!(io,Vector{ComplexF32}(undef,n));eof(io)||error("RA trailer");all(isfinite,a)||error("nonfinite input");return a,dims
 end
end
mutable struct Observer
 measured::Bool
 running::Bool
 elapsed::Float64
 start::UInt64
 trial::Int
 oracle::String
 output::String
 private::String
 prefix::Vector{Float32}
 final::Vector{Float32}
 maxerr::Float64
 maxratio::Float64
 checked::Int
 trace_checked::Int
 lowner
 sowner
end
function pause!(o)
 CUDA.device_synchronize(blocking=true)
 if o.measured && o.running;o.elapsed+=(time_ns()-o.start)*1e-9;o.running=false;end
end
function resume!(o)
 if o.measured;o.start=time_ns();o.running=true;end
end
components(a::Vector{ComplexF32})=reinterpret(Float32,a)
components(a::Vector{Float32})=a
function check_field!(o,name,a;trace=false)
 path=joinpath(o.oracle,(trace ? "trace_iteration3_" : "")*name*".bin")
 filesize(path)==8length(a)||error("oracle bytecount $name")
 ref=open(path,"r") do io;read!(io,Vector{Float64}(undef,length(a)));end
 all(isfinite,ref)||error("nonfinite oracle $name")
 budget=1e-6+.002*maximum(abs,ref;init=0.0)
 for i in eachindex(a)
  isfinite(a[i])||error("nonfinite GPU field $name")
  d=abs(Float64(a[i])-ref[i]);o.maxerr=max(o.maxerr,d);o.maxratio=max(o.maxratio,d/budget)
  d<=budget||error("independent budget $name index$i: $d > $budget")
 end
 trace ? (o.trace_checked+=length(a)) : (o.checked+=length(a))
end
function trace!(o,name,a)
 o.trial==0||return
 pause!(o);v=components(Array(a));write(joinpath(o.private,"warmup_iteration3_"*name*".bin"),v);check_field!(o,name,v;trace=true);resume!(o)
end
function input_observe!(o,img,mask,himg,hmask)
 pause!(o);a=components(Array(img));b=components(Array(mask))
 reinterpret(UInt8,a)==reinterpret(UInt8,himg)&&reinterpret(UInt8,b)==reinterpret(UInt8,hmask)||error("actual device input bytes")
 o.prefix=vcat(a,b);resume!(o)
end
function final_observe!(o,fields,ls,ls2,lu,vt,svt,info,ser0,ser1)
 pause!(o)
 arrays=[(name,copy(components(Array(a)))) for (name,a) in fields]
 push!(arrays,("ser_imgz",Float32[ser0]));push!(arrays,("ser_imgr",Float32[ser1]));push!(arrays,("ls",Array(ls)));push!(arrays,("ls2",copy(components(Array(ls2)))))
 o.final=copy(o.prefix)
 for (_,a) in arrays;append!(o.final,a);end
 for owner in (o.lowner,o.sowner)
  g=components(Array(@view owner[end-16:end]));all(==(12345.25f0),g)||error("outer guard");append!(o.final,g)
 end
 # Persist complete actual state before independent checks so failed executions retain bytes.
 write(joinpath(o.output,"trial$(o.trial).bin"),o.final)
 fac=vcat(components(Array(lu)),components(Array(vt)),components(Array(svt)));write(joinpath(o.private,"trial$(o.trial)_raw_SVD_factors.bin"),fac)
 for (name,a) in arrays;check_field!(o,name,a);end
 only(Array(info))==0||error("SVD convergence info")
 resume!(o)
end
slice(a,k,n)=@view a[k+1:k+n]
function forward!(ctx,out,inp,mask,n,r,c)
 ccall((:tgv_fft,TVLIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint),ctx,raw(out),raw(inp),-1)
 launch(Val(:scaledata),n,out,n,inv(sqrt(Float32(r*c))));launch(Val(:arraydot),n,out,out,mask,n)
end
function backward!(ctx,out,inp,mask,n,r,c)
 tmp=device_array(ComplexF32,n);copy_device!(tmp,inp,n);launch(Val(:arraydot),n,tmp,tmp,mask,n)
 ccall((:tgv_fft,TVLIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint),ctx,raw(out),raw(tmp),1)
 launch(Val(:scaledata),n,out,n,inv(sqrt(Float32(r*c))));free_array(tmp)
end
function gradient!(out,a,n,r,c,t,mu,axis)
 if axis==1
  launch(Val(:grad_xx),n-r*t,out,a,n,r,c,t);launch(Val(:grad_xx_bound),r*t,out,n,r,c,t)
 elseif axis==2
  launch(Val(:grad_yy),n-c*t,out,a,n,r,c,t);launch(Val(:grad_yy_bound),c*t,out,n,r,c,t)
 else
  launch(Val(:grad_tt),n-r*c,out,a,n,r,c,t,mu);launch(Val(:grad_tt_bound),r*c,out,n,r,c,t)
 end
end
function divergence!(out,a,n,r,c,t,mu,axis)
 if axis==1
  launch(Val(:grad_xx_adj),n-2r*t,out,a,n,r,c,t);launch(Val(:grad_xx_adj_zero),r*t,out,a,r,c,t);launch(Val(:grad_xx_adj_bound),r*t,out,a,r,c,t)
 elseif axis==2
  launch(Val(:grad_yy_adj),n-2c*t,out,a,n,r,c,t);launch(Val(:grad_yy_adj_zero),c*t,out,a,r,c,t);launch(Val(:grad_yy_adj_bound),c*t,out,a,r,c,t)
 else
  launch(Val(:grad_tt_adj),n-2r*c,out,a,n,r,c,t,mu);launch(Val(:grad_tt_adj_zero),r*c,out,a,r,c,t);launch(Val(:grad_tt_adj_bound),r*c,out,a,r,c,t)
 end
end
function ser(ctx,a,img,n)
 diff=device_array(Float32,n);tmp=device_array(Float32,n);launch(Val(:arrayabs),n,diff,a,n);launch(Val(:arrayabs),n,tmp,img,n);launch(Val(:arrayadd_real),n,diff,diff,tmp,n,1f0,-1f0)
 imgnorm=ccall((:tgv_norm,TVLIB),Cfloat,(Ptr{Cvoid},Ptr{Cvoid},Int64),ctx,raw(tmp),n);diffnorm=ccall((:tgv_norm,TVLIB),Cfloat,(Ptr{Cvoid},Ptr{Cvoid},Int64),ctx,raw(diff),n)
 value=-20f0*log10(diffnorm/imgnorm);free_array(diff);free_array(tmp);value
end
function vendor_buffer(ctx,i,::Type{T},n)where T
 p=ccall((:tgv_buffer,TVLIB),Ptr{Cvoid},(Ptr{Cvoid},Cint),ctx,i);p==C_NULL&&error("vendor buffer");unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(n,);own=false)
end
function pipeline!(o,l,s,himg,hmask,shape,iterations)
 r,c,t=shape;n=r*c*t;mu=1f0;tau=.25f0;sigma=.25f0;beta=.5f0
 ctx=ccall((:tgv_create,TVLIB),Ptr{Cvoid},())
 img=device_array(ComplexF32,n);h2d!(img,himg);mask=device_array(ComplexF32,n);h2d!(mask,hmask);input_observe!(o,img,mask,himg,hmask)
 mx=ccall((:tgv_normalizer,TVLIB),Cfloat,(Ptr{Cvoid},Ptr{Cvoid},Int64),ctx,raw(img),n);launch(Val(:scaledata),n,img,n,inv(mx))
 z=device_array(ComplexF32,n);b=device_array(ComplexF32,n);ccall((:tgv_fft_init,TVLIB),Cvoid,(Ptr{Cvoid},Cint,Cint,Cint),ctx,r,c,t);forward!(ctx,b,img,mask,n,r,c);backward!(ctx,z,b,mask,n,r,c)
 w=device_array(ComplexF32,3n);p=device_array(ComplexF32,3n);q=device_array(ComplexF32,6n);dualr=device_array(ComplexF32,n);lb=device_array(ComplexF32,n);sb=device_array(ComplexF32,n);wb=device_array(ComplexF32,3n);lo=device_array(ComplexF32,n);so=device_array(ComplexF32,n);wo=device_array(ComplexF32,3n);tmp=device_array(ComplexF32,12n)
 launch(Val(:arrayreal),n,l,z,n);zero_device!(s);zero_device!(w);copy_device!(sb,s,n);copy_device!(lb,l,n);copy_device!(wb,w,3n);zero_device!(p);zero_device!(q);zero_device!(dualr);zero_device!(tmp)
 alpha0=sqrt(2f0)*.004f0;alpha1=.004f0
 ccall((:tgv_svd_init,TVLIB),Cvoid,(Ptr{Cvoid},Cint,Cint),ctx,r*c,t)
 lu=vendor_buffer(ctx,0,ComplexF32,n);ls=vendor_buffer(ctx,1,Float32,t);ls2=vendor_buffer(ctx,2,ComplexF32,t);vt=vendor_buffer(ctx,3,ComplexF32,t*t);svt=vendor_buffer(ctx,4,ComplexF32,t*t);info=vendor_buffer(ctx,5,Int32,1)
 for it in 1:iterations
  a0=ccall((:tgv_alpha,TVLIB),Cfloat,(Cfloat,Cfloat,Cint,Cint),alpha0,alpha0*.5f0,iterations,it);a1=ccall((:tgv_alpha,TVLIB),Cfloat,(Cfloat,Cfloat,Cint,Cint),alpha1,alpha1*.5f0,iterations,it)
  copy_device!(lo,l,n);copy_device!(so,s,n);copy_device!(wo,w,3n)
  launch(Val(:arrayadd),n,tmp,lb,sb,n,1f0,1f0);forward!(ctx,tmp,tmp,mask,n,r,c);launch(Val(:update_r),n,dualr,tmp,b,n,sigma)
  it==3&&trace!(o,"r",dualr)
  for axis in 1:3;gradient!(slice(tmp,(axis-1)*n,n),sb,n,r,c,t,mu,axis);end
  launch(Val(:proj_p),n,p,tmp,wb,n,sigma,a1);it==3&&trace!(o,"p",p)
  for (k,(component,axis)) in enumerate(((0,1),(1,2),(2,3),(0,2),(1,1),(2,1),(0,3),(1,3),(2,2)))
   divergence!(slice(tmp,(k+2)*n,n),slice(wb,component*n,n),n,r,c,t,mu,axis)
  end
  launch(Val(:proj_q),n,q,tmp,n,sigma,a0);it==3&&trace!(o,"q",q)
  backward!(ctx,z,dualr,mask,n,r,c)
  for axis in 1:3;divergence!(slice(tmp,(axis-1)*n,n),slice(p,(axis-1)*n,n),n,r,c,t,mu,axis);end
  launch(Val(:update_s),n,s,tmp,z,n,tau);it==3&&trace!(o,"s",s)
  for (k,(component,axis)) in enumerate(((0,1),(3,2),(4,3),(3,1),(1,2),(5,3),(4,1),(5,2),(2,3)))
   gradient!(slice(tmp,(k+2)*n,n),slice(q,component*n,n),n,r,c,t,mu,axis)
  end
  launch(Val(:update_w),n,w,tmp,p,n,tau);it==3&&trace!(o,"w",w)
  launch(Val(:arrayadd),n,l,l,z,n,1f0,-tau);ccall((:tgv_svd,TVLIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Cint,Cint),ctx,raw(l),r*c,t);launch(Val(:shrink),t,ls2,ls,beta,t);ccall((:tgv_reconstruct,TVLIB),Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Cint,Cint),ctx,raw(l),r*c,t);it==3&&trace!(o,"l",l)
  launch(Val(:arrayadd),n,sb,s,so,n,2f0,-1f0);launch(Val(:arrayadd),3n,wb,w,wo,3n,2f0,-1f0);launch(Val(:arrayadd),n,lb,l,lo,n,2f0,-1f0)
  if it==3;trace!(o,"lb",lb);trace!(o,"sb",sb);trace!(o,"wb",wb);trace!(o,"tmp",tmp);end
 end
 imgr=device_array(ComplexF32,n);launch(Val(:arrayadd),n,imgr,l,s,n,1f0,1f0);backward!(ctx,z,b,mask,n,r,c);ser0=ser(ctx,z,img,n);ser1=ser(ctx,imgr,img,n)
 final_observe!(o,(("img",img),("mask",mask),("imgz",z),("imgb",b),("imgl",l),("imgs",s),("w",w),("p",p),("q",q),("r",dualr),("lbar",lb),("sbar",sb),("wbar",wb),("lold",lo),("sold",so),("wold",wo),("tmp",tmp),("imgr",imgr)),ls,ls2,lu,vt,svt,info,ser0,ser1)
 for a in (img,b,z,imgr,mask,w,p,q,dualr,lb,sb,wb,lo,so,wo,tmp);free_array(a);end
 ccall((:tgv_source_destroy,TVLIB),Cvoid,(Ptr{Cvoid},),ctx)
 pause!(o);ccall((:tgv_finish_destroy,TVLIB),Cvoid,(Ptr{Cvoid},),ctx)
end
function study_main(model)
 length(ARGS)==5||error("oracle image.ra mask.ra iterations private_root")
 CUDA.device!(0);CUDA.stream!(CUDA.legacy_stream());image,shape=read_ra(ARGS[2]);mask,ms=read_ra(ARGS[3]);shape==ms||error("mask shape");n=prod(shape);it=parse(Int,ARGS[4]);all(>=(3),shape)&&shape[1]*shape[2]>=shape[3]&&it>=3||error("geometry")
 ccall((:tgv_resolution,TVLIB),Cvoid,())
 lo=device_array(ComplexF32,n+17);so=device_array(ComplexF32,n+17);l=@view lo[1:n];s=@view so[1:n];guard=fill(ComplexF32(12345.25,12345.25),17);GC.@preserve guard begin;h2d!(@view(lo[n+1:n+17]),guard);h2d!(@view(so[n+1:n+17]),guard);CUDA.device_synchronize(blocking=true);end
 out=get(ENV,"STUDY_OUTPUT_DIR","");isempty(out)&&error("STUDY_OUTPUT_DIR");private=joinpath(ARGS[5],"pid$(getpid())");mkpath(private)
 for trial in 0:5
  o=Observer(trial>0,false,0.,0,trial,ARGS[1],out,private,Float32[],Float32[],0.,0.,0,0,lo,so)
  CUDA.device_synchronize(blocking=true);resume!(o)
  GC.@preserve image mask pipeline!(o,l,s,image,mask,shape,it)
  pause!(o)
  if trial==0;println("UNTIMED_TRACE {\"actual_third_iteration\":true,\"checked_components\":$(o.trace_checked),\"full_trajectory_iterations\":$it}");continue;end
  artifact=joinpath(out,"trial$trial.bin")
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_independent_field_budget_ratio\":%.17g,\"jit_excluded\":true,\"timing_valid\":true,\"input_contract_matched\":true,\"shared_inputs\":true,\"timed_region_matched\":true,\"timing_region\":\"source_tgv_cs_completed_pipeline_excluding_observers_and_formatting\",\"private_output_directory\":\"%s\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,o.elapsed,length(o.final),o.maxerr,o.maxratio,private,artifact,length(o.final));flush(stdout)
 end
 free_array(lo);free_array(so)
end
