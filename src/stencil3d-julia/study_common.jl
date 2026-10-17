const NS=30;const VOL=NS^3;const MAT=((2.,.125,.25),(.125,3.,.375),(.25,.375,4.))
@inline sv(s,plane,y,z)=@inbounds s[plane*Int32(256)+y*Int32(16)+z+1]
@inline function tensor_dot(sigma,axis,site,vol,xd,yd,zd)
 @inbounds return sigma[axis*Int32(3)*vol+site+1]*xd+sigma[(axis*Int32(3)+Int32(1))*vol+site+1]*yd+sigma[(axis*Int32(3)+Int32(2))*vol+site+1]*zd
end
@inline function xface(s,c,n,y,z,sigma,site,vol)
 xd=-sv(s,c,y,z)+sv(s,n,y,z);yd=(-sv(s,c,y-1,z)+sv(s,c,y+1,z)-sv(s,n,y-1,z)+sv(s,n,y+1,z))/4.;zd=(-sv(s,c,y,z-1)+sv(s,c,y,z+1)-sv(s,n,y,z-1)+sv(s,n,y,z+1))/4.
 return tensor_dot(sigma,Int32(0),site,vol,xd,yd,zd)
end
@inline function yface(s,p,c,n,y,z,sigma,site,vol)
 xd=(-sv(s,p,y,z)-sv(s,p,y+1,z)+sv(s,n,y,z)+sv(s,n,y+1,z))/4.;yd=-sv(s,c,y,z)+sv(s,c,y+1,z);zd=(-sv(s,c,y,z-1)+sv(s,c,y,z+1)-sv(s,c,y+1,z-1)+sv(s,c,y+1,z+1))/4.
 return tensor_dot(sigma,Int32(1),site,vol,xd,yd,zd)
end
@inline function zface(s,p,c,n,y,z,sigma,site,vol)
 xd=(-sv(s,p,y,z)-sv(s,p,y,z+1)+sv(s,n,y,z)+sv(s,n,y,z+1))/4.;yd=(-sv(s,c,y-1,z)-sv(s,c,y-1,z+1)+sv(s,c,y+1,z)+sv(s,c,y+1,z+1))/4.;zd=-sv(s,c,y,z)+sv(s,c,y,z+1)
 return tensor_dot(sigma,Int32(2),site,vol,xd,yd,zd)
end
function direct_oracle(psi,sigma,n)
 result=zeros(Float64,n^3);v=n^3;at(x,y,z)=psi[z+n*(y+n*x)+1]
 function face(axis,x,y,z)
  if axis==0
   xd=at(x,y,z)-at(x-1,y,z);yd=(-at(x-1,y-1,z)+at(x-1,y+1,z)-at(x,y-1,z)+at(x,y+1,z))/4;zd=(-at(x-1,y,z-1)+at(x-1,y,z+1)-at(x,y,z-1)+at(x,y,z+1))/4
  elseif axis==1
   xd=(-at(x-1,y-1,z)-at(x-1,y,z)+at(x+1,y-1,z)+at(x+1,y,z))/4;yd=at(x,y,z)-at(x,y-1,z);zd=(-at(x,y-1,z-1)+at(x,y-1,z+1)-at(x,y,z-1)+at(x,y,z+1))/4
  else
   xd=(-at(x-1,y,z-1)-at(x-1,y,z)+at(x+1,y,z-1)+at(x+1,y,z))/4;yd=(-at(x,y-1,z-1)-at(x,y-1,z)+at(x,y+1,z-1)+at(x,y+1,z))/4;zd=at(x,y,z)-at(x,y,z-1)
  end
  i=z+n*(y+n*x)+1;return sigma[(axis*3)*v+i]*xd+sigma[(axis*3+1)*v+i]*yd+sigma[(axis*3+2)*v+i]*zd
 end
 for x in 1:n-2,y in 1:n-2,z in 1:n-2
  result[z+n*(y+n*x)+1]=face(0,x+1,y,z)-face(0,x,y,z)+face(1,x,y+1,z)-face(1,x,y,z)+face(2,x,y,z+1)-face(2,x,y,z)
 end
 return result
end
function study(run!)
 sigma=Float64[MAT[axis+1][dir+1]*(1+mod(i,7)/16) for axis in 0:2 for dir in 0:2 for i in 0:VOL-1];fields=(Float64[(mod(i,19)-9)/8 for i in 0:VOL-1],fill(1.25,VOL));S=CuArray(sigma);inputs=map(CuArray,fields);Y=CUDA.zeros(Float64,VOL);template=CUDA.zeros(Float64,1024);expected=map(psi->direct_oracle(psi,sigma,NS),fields)
 for x in inputs;run!(x,Y,S,template,Int32(NS));end;CUDA.synchronize()
 for trial in 1:5
  total=0.;got=Float64[];want=Float64[]
  for case in 1:2
   fill!(Y,0.);CUDA.synchronize();start=time_ns();for repeat in 1:10;run!(inputs[case],Y,S,template,Int32(NS));end;CUDA.synchronize();total+=(time_ns()-start)*1e-9/10;append!(got,Array(Y));append!(want,expected[case])
  end
  err=maximum(abs.(got.-want));ok=all(isfinite,got)&&err<=1e-10;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":54000,\"max_abs_error\":",err,",\"timing_region\":\"sum_nonuniform_and_constant_anisotropic_stencil_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":54000,\"byte_order\":\"little\"}}")
  ok||error("Anisotropicfaceflux directCPUoracle failed")
 end
end
