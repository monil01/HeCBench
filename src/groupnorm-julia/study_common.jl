using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function norm_math(x,w,b,dy)
 n=32768;mean=zeros(Float64,32);rs=zeros(Float64,32);out=zeros(Float64,n);dx=zeros(Float64,n);dw=zeros(Float64,32);db=zeros(Float64,32)
 for block in 0:31
  base=block*1024;wb=mod(block,8)*4;values=@view x[base+1:base+1024];mu=sum(values)/1024;variance=sum(abs2,values)/1024-mu^2;s=inv(sqrt(variance+Float64(1.f-5)));mean[block+1]=mu;rs[block+1]=s
  first=0.;second=0.
  for i in 0:1023
   ch=wb+div(i,256)+1;norm=(x[base+i+1]-mu)*s;out[base+i+1]=norm*w[ch]+b[ch];wd=w[ch]*dy[base+i+1];first+=wd;second+=wd*norm;dw[ch]+=dy[base+i+1]*norm;db[ch]+=dy[base+i+1]
  end
  for i in 0:1023
   ch=wb+div(i,256)+1;norm=(x[base+i+1]-mu)*s;dx[base+i+1]=(w[ch]*dy[base+i+1]-first/1024-norm*second/1024)*s
  end
 end
 out,mean,rs,dx,dw,db
end
function groupnorm_study(factory)
 mode=isempty(ARGS) ? "forward" : ARGS[1];n=32768
 hx=Float32[(mod(i*13+div(i,257)*7,127)-63)/64 for i in 0:n-1];hw=Float32[.5+mod(c,7)/16 for c in 0:31];hb=Float32[(mod(c*3,9)-4)/32 for c in 0:31];hdy=Float32[(mod(i*11+div(i,131),31)-15)/32 for i in 0:n-1]
 xd=Float64.(hx);wd=Float64.(hw);bd=Float64.(hb);dyd=Float64.(hdy);eo,em,er,edx,edw,edb=norm_math(xd,wd,bd,dyd)
 # Independently check the analytic input derivative by centered finite differences.
 delta=1e-4
 for index in [1,256,257,1024,1025,16384,32512,32768]
  xp=copy(xd);xm=copy(xd);xp[index]+=delta;xm[index]-=delta
  yp=norm_math(xp,wd,bd,dyd)[1];ym=norm_math(xm,wd,bd,dyd)[1]
  numeric=sum(dyd.*(yp.-ym))/(2delta)
  abs(numeric-edx[index])<1e-5||error("finite difference input gradient failed")
 end
 x=CuArray(hx);w=CuArray(hw);b=CuArray(hb);dy=CuArray(hdy);out=CUDA.zeros(Float32,n);mean=CuArray(Float32.(em));rs=CuArray(Float32.(er));dx=CUDA.zeros(Float32,n);dw=CUDA.zeros(Float32,32);db=CUDA.zeros(Float32,32)
 operation! =factory(x,w,b,out,mean,rs,dy,dx,dw,db,mode)
 reset! =()->begin;fill!(out,0.f0);fill!(dx,0.f0);fill!(dw,0.f0);fill!(db,0.f0);copyto!(mean,Float32.(em));copyto!(rs,Float32.(er));end
 backward=mode=="backward";expected=Float32.(backward ? vcat(edx,edw,edb) : vcat(eo,em,er));readback=backward ? ()->vcat(Array(dx),Array(dw),Array(db)) : ()->vcat(Array(out),Array(mean),Array(rs))
 println("STUDY_CONTRACT mode=",mode," B4_C32_H16_W16_groups8 Float32 full32832observables atol1e-3_stricter_than_original_out_dw_db_1e-2_dx1.0 independentFloat64formula_and8finite_differences;backward_fixedcanonicalstats_1repeat forward10")
 OwnedStudy.run_trials!(operation!,reset!,readback,expected,backward ? 1 : 10;atol=1e-3,rtol=0.)
end
