function fixtures()
 out=[]
 for(ni,nj,nk) in ((13,11,18),(19,13,20)),mode in 0:2
  nkdim=nk+6;n=ni*nj*nkdim;bounds=Int32[-2,ni-3,-2,nj-3,-2,nk+3];u=zeros(Float64,3n);mu=fill(2.,n);la=fill(3.,n);met=zeros(Float64,4n);met[1:n].=1.;met[3n+1:4n].=1.;jac=ones(Float64,n);init=Float64[(mod(i*7,31)-15)/32 for i in 0:3n-1];sx=ones(Float64,ni);sy=ones(Float64,nj);cof=Float64[i/1000 for i in 0:863]
  for iz in 0:nkdim-1,iy in 0:nj-1,ix in 0:ni-1
   q=ix+ni*(iy+nj*iz);x=ix-2;y=iy-2;z=iz-2
   if mode==0;u[q+1]=x*x+2x*y+3x*z;u[n+q+1]=2y*y+4x*y+5y*z;u[2n+q+1]=3z*z+6x*z+7y*z
   elseif mode==1;u[q+1]=1;u[n+q+1]=2;u[2n+q+1]=3
   else
    for c in 0:2;u[c*n+q+1]=(mod(q*17+c*13,97)-48)/16;end
    mu[q+1]+=mod(q,7)/32;la[q+1]+=mod(q,5)/32;jac[q+1]+=mod(q,13)/64;met[q+1]+=mod(q,5)/64;met[n+q+1]=.125+mod(q,3)/128;met[2n+q+1]=.0625+mod(q,7)/128;met[3n+q+1]=.75+mod(q,11)/128
   end
  end
  if mode==2;sx.=[1+i/64 for i in 0:ni-1];sy.=[1+i/64 for i in 0:nj-1];end
  push!(out,(ni,nj,nk,mode,bounds,u,mu,la,met,jac,init,sx,sy,cof))
 end
 out
end
function stageargs(k,bd,nk)
 fi,li,fj,lj,fk,lk=bd
 if k==1;st=(fi+2,fj+2,Int32(1));en=(li-1,lj-1,Int32(7));block=(16,4,3)
 elseif k==5;st=(fi+2,fj+2,Int32(nk-5));en=(li-1,lj-1,Int32(nk+1));block=(16,4,4)
 else;st=(fi+2,fj+2,Int32(7));en=(li-1,lj-1,Int32(nk-5));block=(64,2,2)
 end
 head=(st[1],en[1],st[2],en[2],st[3],en[3],bd...);return head,block
end
function cpu_reference(case)
 ni,nj,nk,mode,bd,u,mu,la,met,jac,init,sx,sy,cof=case;lu=copy(init);ac=view(cof,469:852);bo=view(cof,415:462);gh=view(cof,853:858);arrays=(u,mu,la,met,jac,lu,ac,bo,gh,ac,gh,sx,sy);functions=(cpu_kernel1!,cpu_kernel2!,cpu_kernel3!,cpu_kernel4!,cpu_kernel5!)
 for k in 1:5
  head,block=stageargs(k,bd,nk);s0,e0,s1,e1,s2,e2,rest...=head
  for z in s2:e2-1,y in s1:e1-1,x in s0:e0-1
   args=(x,e0,y,e1,z,e2,bd...);if k==5;functions[k](args...,Int32(nk),1.,-1.,arrays...);else;functions[k](args...,1.,-1.,arrays...);end
  end
 end
 return lu
end
function physical_check(case,lu)
 ni,nj,nk,mode,bd,u,mu,la,met,jac,init,sx,sy,cof=case;mode==2&&return true;n=ni*nj*(nk+6);err=0.
 for z in 7:nk-6,y in 0:nj-5,x in 0:ni-5,c in 0:2
  q=(x+2)+ni*((y+2)+nj*(z+2))+c*n+1;expected=init[q]-(mode==0 ? (64.,73.,82.)[c+1] : 0.);err=max(err,abs(lu[q]-expected))
 end
 return err<=1e-8
end
function study(allocate,launch)
 cases=[];prefix=Float64[]
 for case in fixtures()
  ni,nj,nk,mode,bd,u,mu,la,met,jac,init,sx,sy,cof=case;h=(u,mu,la,met,jac,init,sx,sy,cof);a=map(x->begin d=allocate(Float64,length(x));copyto!(d,x);d end,h);ref=cpu_reference(case);append!(prefix,Float64[ni,nj,nk,mode,3]);append!(prefix,Float64.(bd));for v in a;append!(prefix,Array(v));end;push!(cases,(case,a,ref))
 end
 function run(item)
  case,a,ref=item;ni,nj,nk,mode,bd,u,mu,la,met,jac,init,sx,sy,cof=case;ud,mud,lad,metd,jacd,lu,sxd,syd,cofd=a;ac=view(cofd,469:852);bo=view(cofd,415:462);gh=view(cofd,853:858);arr=(ud,mud,lad,metd,jacd,lu,ac,bo,gh,ac,gh,sxd,syd);time=0.
  for rep in 1:3
   copyto!(lu,init);CUDA.synchronize(CUDA.context());start=time_ns();launch(bd,nk,arr);CUDA.synchronize(CUDA.context());time+=(time_ns()-start)*1e-9/3
  end
  v=Array(lu);err=maximum(abs.(v.-ref));ok=all(isfinite,v)&&all(abs.(v.-ref).<=1e-8.+1e-10.*abs.(ref))&&physical_check(case,v);return v,time,ok,err
 end
 for c in cases;run(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;err=0.;ts=Float64[]
  for c in cases;v,t,pass,e=run(c);append!(out,v);elapsed+=t;ok&=pass;err=max(err,e);push!(ts,t);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(ts,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum6_three_original_fivekernel_pipeline_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("SW4FullOutputOrManufacturedNavier")
 end
end
