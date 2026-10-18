function oracle(M,g,f,v,n)
 a=zeros(UInt8,n);c=zeros(UInt8,n);t=zeros(UInt8,n)
 function interp(f,x,y,z)
  ix=floor(Int,x);iy=floor(Int,y);iz=floor(Int,z);dx=Float64(x)-ix;dy=Float64(y)-iy;dz=Float64(z)-iz;ans=0.
  for oz in 0:1,oy in 0:1,ox in 0:1
   j=ix+ox+v*(iy+oy-1+v*(iz+oz-1));ans+=Float64(f[j])*(ox==0 ? 1-dx : dx)*(oy==0 ? 1-dy : dy)*(oz==0 ? 1-dz : dz)
  end
  UInt8(floor(ans+0.5))
 end
 for i in 0:n-1
  r=RAN[mod(i,97)+1];rx=Float32(mod(i,v-2))+1f0+r;ry=Float32(mod(i÷(v-2),v-2))+1f0+r;rz=floor(Float32(i)/Float32(v-2))/Float32(v-2)+1f0+r
  xp=muladd(M[9],rz,muladd(M[5],ry,M[1]*rx))+M[13];yp=muladd(M[10],rz,muladd(M[6],ry,M[2]*rx))+M[14];zp=muladd(M[11],rz,muladd(M[7],ry,M[3]*rx))+M[15]
  if 1<=xp<v&&1<=yp<v&&1<=zp<v;a[i+1]=interp(f,xp,yp,zp);c[i+1]=interp(g,rx,ry,rz);t[i+1]=1;end
 end
 return a,c,t
end
function study(allocate,launch)
 cases=[];prefix=Float32[]
 for v in (17,33),mode in 0:1
  n=v^3;size=(v+1)^2*(v+5);g=UInt8[mod(i*17+i÷v*7,251)for i in 0:size-1];f=UInt8[mod(i*29+i÷(v*v)*13,251)for i in 0:size-1];M=zeros(Float32,16);M[[1,6,11,16]].=1f0;if mode==1;M[13]=0.25f0;M[14]=0.5f0;M[15]=0.125f0;end
  m=allocate(Float32,16);gd=allocate(UInt8,size);fd=allocate(UInt8,size);a=allocate(UInt8,n);c=allocate(UInt8,n);t=allocate(UInt8,n);copyto!(m,M);copyto!(gd,g);copyto!(fd,f);ref=oracle(M,g,f,v,n)
  append!(prefix,Float32[v,mode,n,size,3]);append!(prefix,Array(m));append!(prefix,Float32.(Array(gd)));append!(prefix,Float32.(Array(fd)));push!(cases,(m,n,gd,fd,v,a,c,t,ref))
 end
 function run(cas)
  m,n,g,f,v,a,c,t,ref=cas;CUDA.synchronize(CUDA.context());start=time_ns()
  for rep in 1:3;launch(m,n,g,f,v,a,c,t);end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/3;aa=Array(a);cc=Array(c);tt=Array(t);err=max(maximum(abs.(Int.(aa).-Int.(ref[1]))),maximum(abs.(Int.(cc).-Int.(ref[2]))));ok=err==0&&tt==ref[3];return vcat(Float32.(aa),Float32.(cc),Float32.(tt)),elapsed,ok,err
 end
 for c in cases;run(c);end
 for trial in 1:5
  out=copy(prefix);time=0.;ok=true;err=0;ts=Float64[]
  for c in cases;a,t,pass,e=run(c);append!(out,a);time+=t;ok&=pass;err=max(err,e);push!(ts,t);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(ts,","));println("STUDY_TRIAL {"trial":",trial,","time_seconds":",time,","correct":",ok,","checked_values":",length(out),","max_abs_error":",err,","timing_region":"sum4_three_spm_kernel_means","jit_excluded":true,"internal_repeats":3,"output_artifact":{"path":"",path,"","dtype":"float32","count":",length(out),","byte_order":"little"}}");ok||error("SPMFullOutputs")
 end
end
