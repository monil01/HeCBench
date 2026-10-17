using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function hacc_study(factory)
 n1=784;n2=15000;v=collect(reinterpret(Float32,read(joinpath(@__DIR__,"study_input_f32.bin"))));length(v)==4n2||error("fixture size")
 hx=v[1:n2];hy=v[n2+1:2n2];hz=v[2n2+1:3n2];hm=v[3n2+1:4n2]
 coeff=Float32[.269327,-.0750978,.0114808,-.00109313,.0000605491,-.00000147177];fc=.23f0;cut=.5f0;soft=.03f0
 expected=zeros(Float64,3n1)
 for i in 1:n1
  ax=0.;ay=0.;az=0.
  for j in 1:n2
   dx=Float64(hx[j])-Float64(hx[i]);dy=Float64(hy[j])-Float64(hy[i]);dz=Float64(hz[j])-Float64(hz[i]);r2=dx*dx+dy*dy+dz*dz
   m=r2<Float64(cut) ? Float64(hm[j]) : 0.;f=r2+Float64(soft)
   poly=Float64(coeff[6]);for k in 5:-1:1;poly=Float64(coeff[k])+r2*poly;end
   force=m*(1/(f*sqrt(f))-poly);ax+=force*dx;ay+=force*dy;az+=force*dz
  end
  expected[i]=ax*Float64(fc);expected[n1+i]=ay*Float64(fc);expected[2n1+i]=az*Float64(fc)
 end
 x=CuArray(hx);y=CuArray(hy);z=CuArray(hz);mass=CuArray(hm);vx=CUDA.zeros(Float32,n1);vy=CUDA.zeros(Float32,n1);vz=CUDA.zeros(Float32,n1)
 operation! =factory(x,y,z,mass,vx,vy,vz,n1,n2,cut,soft,fc)
 reset! =()->begin;fill!(vx,0.f0);fill!(vy,0.f0);fill!(vz,0.f0);end
 readback=()->begin
  got=vcat(Array(vx),Array(vy),Array(vz));maxrelative=maximum(abs.((Float64.(got).-expected)./max.(abs.(expected),1.)))
  println("STUDY_RELATIVE_FLOAT64_ERROR ",maxrelative);maxrelative<=1e-4||error("independent relative criterion")
  got
 end
 println("STUDY_CONTRACT source_n1=784_n2=15000 grid4_block256 full2352activecomponents sourceeps1 independentFloat64relative1e-4 one_original_reset_updatepertrial tailhostzeros_notinflated")
 OwnedStudy.run_trials!(operation!,reset!,readback,expected,1;atol=1.,rtol=0.)
end
