using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function lomb_study(factory)
 ns=1000;nf=100000;host=Vector{Float32}(undef,2ns+nf+1);open(joinpath(@__DIR__,"study_input_f32.bin"))do io;read!(io,host);end
 x=host[1:ns];y=host[ns+1:2ns];freqs=host[2ns+1:2ns+nf];scale=host[end];expected=Vector{Float64}(undef,nf);open(joinpath(@__DIR__,"study_expected_f64.bin"))do io;read!(io,expected);end
 energybound=0.5*Float64(scale)*sum(abs2,Float64.(y));blocks=cld(nf,256)*256
 dx=CuArray(x);dy=CuArray(y);df=CuArray(freqs);out=CUDA.zeros(Float32,nf);op=factory(dx,dy,df,out,scale,ns,nf,blocks)
 readback=()->begin
  got=Array(out);all(isfinite,got)&&all(got.>=0)||error("invalidpower");peak=argmax(got);abs(freqs[peak]-1.f0)<0.002f0||error("knownsignalpeak");maximum(got)<=energybound+0.1||error("projectionenergybound")
  println("STUDY_SPECTRAL_CHECKS peak_angular_frequency=",freqs[peak]," maximum_power=",maximum(got)," energy_projection_bound=",energybound," actual_scale=",scale," not_energy_normalized source_blocks=",blocks," active_blocks=",cld(nf,256)," idle_blocks=",blocks-cld(nf,256))
  got
 end
 OwnedStudy.run_trials!(op,()->fill!(out,0.f0),readback,expected,10;atol=1e-1,rtol=0.)
end
