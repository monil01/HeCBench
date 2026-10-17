using CUDA,Printf
function md_fixture(mode)
 n=129;p=[zeros(Float32,n) for _=1:3];neighbors=Vector{Int32}(undef,n*128)
 for i=0:n-1
  x=Float32(i%7)*1.25f0;y=Float32((i÷7)%7)*1.25f0;z=Float32(i÷49)*1.25f0
  if i==1;x=0f0;y=0f0;z=0f0;end
  if i==2;x=3f0;y=1.5f0;z=1.5f0;end
  if mode==1;x+=.5f0;y-=.25f0;z+=1f0;end
  if mode==2;x,y=-y,x;end
  p[1][i+1]=x;p[2][i+1]=y;p[3][i+1]=z;slot=0
  for j=0:n-1
   if j!=i;neighbors[slot*n+i+1]=Int32(j);slot+=1;end
  end
 end
 return p,neighbors
end
function md_independent(p)
 n=length(p[1]);gold=zeros(Float64,3,n);active=0;coincident=0;exactcut=0
 for i=1:n-1,j=i+1:n
  delta=[Float64(p[k][i])-Float64(p[k][j]) for k=1:3];q=sum(abs2,delta);coincident+=q==0;exactcut+=q==13.5
  if 0<q<13.5
   active+=1;factor=1.5*q^-7-2*q^-4
   for k=1:3;v=delta[k]*factor;gold[k,i]+=v;gold[k,j]-=v;end
  end
 end
 active>0&&coincident>0&&exactcut>0||error("Branch fixture incomplete")
 @printf("MD_INPUT active=%d coincident=%d exact_cutoff=%d\n",active,coincident,exactcut)
 return vcat((gold[k,:] for k=1:3)...)
end
function md_source_cpu(p,neighbors)
 n=length(p[1]);out=[zeros(Float32,n) for _=1:3]
 for i=1:n,j=0:127
  other=Int(neighbors[j*n+i])+1;dx=p[1][i]-p[1][other];dy=p[2][i]-p[2][other];dz=p[3][i]-p[3][other];q=dx*dx+dy*dy+dz*dz
  if 0f0<q<13.5f0
   inv=1f0/q;inv6=inv*inv*inv;f=inv*inv6*(1.5f0*inv6-2f0)
   out[1][i]+=dx*f;out[2][i]+=dy*f;out[3][i]+=dz*f
  end
 end
 return vcat(out...)
end
function md_study(launch!)
 n=129;prefix=Float32[];states=[];gold=Float64[];source=Float32[]
 for mode=0:2
  p,neighbors=md_fixture(mode);append!(prefix,vcat(p...));append!(prefix,Float32.(neighbors));g=md_independent(p);append!(gold,g);append!(source,md_source_cpu(p,neighbors));dp=map(CuArray,p);f=[CUDA.zeros(Float32,n) for _=1:3];dn=CuArray(neighbors);push!(states,(p,dp,f,dn));launch!(dp,f,dn,Int32(n))
 end
 potential(r)=1.5/(12*r^12)-2/(6*r^6);r=1.25;h=1e-5;fd=-(potential(r+h)-potential(r-h))/(2*h);analytic=1.5/r^13-2/r^7;abs(fd-analytic)<=1e-8||error("Pairpotentialgradient")
 @printf("MD_PAIR_GRADIENT error=%.12g\n",abs(fd-analytic));CUDA.synchronize();allok=true
 for trial=1:5
  for (_,_,f,_) in states,arr in f;fill!(arr,Float32(NaN));end
  CUDA.synchronize();start=time_ns()
  for rep=1:20,(_,dp,f,dn) in states;launch!(dp,f,dn,Int32(n));end
  CUDA.synchronize();sec=(time_ns()-start)*1e-9/20;vals=Float32[]
  for (_,_,f,_) in states;append!(vals,vcat(map(Array,f)...));end
  science_error=maximum(abs.(Float64.(vals).-gold));science_ok=all(abs(Float64(v)-g)<=1e-5+1e-6*abs(g) for (v,g) in zip(vals,gold));err=maximum(abs.(vals.-source));momentum=maximum(abs(sum(Float64.(vals[(mode*3+axis)*n+1:(mode*3+axis+1)*n]))) for mode=0:2 for axis=0:2)
  # Translation and ninety-degree rotation are independent geometric consistency checks.
  f0=reshape(vals[1:3*n],n,3);f1=reshape(vals[3*n+1:6*n],n,3);f2=reshape(vals[6*n+1:9*n],n,3)
  translation=maximum(abs.(f1.-f0));rotation=maximum(abs.(f2.-hcat(-f0[:,2],f0[:,1],f0[:,3])))
  normalized_momentum=momentum/maximum(sum(abs,Float64.(vals[(mode*3+axis)*n+1:(mode*3+axis+1)*n])) for mode=0:2 for axis=0:2)
  ok=all(isfinite,vals)&&err<=1e-5&&science_ok&&normalized_momentum<=1e-6&&translation<=1e-5&&rotation<=1e-5;allok&=ok
  @printf("MD_SCIENCE trial=%d pairwise_error=%.12g net_force=%.12g translation=%.12g rotation=%.12g\n",trial,science_error,momentum,translation,rotation)
  out=vcat(prefix,vals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":1161,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"net_force_inf\":%.12g,\"independent_pair_error_inf\":%.12g,\"normalized_net_force\":%.12g,\"translation_error_inf\":%.12g,\"rotation_error_inf\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":20,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",err,momentum,science_error,normalized_momentum,translation,rotation,path,length(out))
 end
 allok||error("MDfullforce/independentphysics mismatch");println("PASS")
end
