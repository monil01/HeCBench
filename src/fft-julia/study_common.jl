using CUDA
include(joinpath(@__DIR__,"study_fft_math.jl"))
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function fft_study(factory)
 nfft=128;n=nfft*512;repeats=10
 input=ComplexF64[ComplexF64(Float64(mod(17*j+13*f,257)-128)/128,Float64(mod(29*j+7*f,251)-125)/128) for f in 0:nfft-1 for j in 0:511]
 twiddle=ComplexF64[ComplexF64(cos(-2*pi*j/512),sin(-2*pi*j/512)) for j in 0:511];spectrum=similar(input)
 for f in 0:nfft-1,k in 0:511
  sum=ComplexF64(0)
  for j in 0:511;sum+=input[f*512+j+1]*twiddle[mod(k*j,512)+1];end
  spectrum[f*512+k+1]=sum
 end
 expected=vcat(collect(reinterpret(Float64,spectrum)),collect(reinterpret(Float64,input)))
 data=CuArray(input);forward!,inverse! = factory(data,nfft)
 launch! = ()->begin;forward!();inverse!();end
 readback=()->begin
  roundtrip=Array(data)
  forward!();CUDA.synchronize();freq=Array(data)
  original_energy=sum(abs2,input);fft_energy=sum(abs2,freq);energy_error=abs(fft_energy-512*original_energy)/(512*original_energy)
  energy_error<1e-10||error("Parsevalenergy")
  println("STUDY_FFT_ACCURACY forward_and_roundtrip_all_components=true energy_relative_error=",energy_error," storage=ComplexF64 native_tolerance=1e-6")
  vcat(collect(reinterpret(Float64,freq)),collect(reinterpret(Float64,roundtrip)))
 end
 OwnedStudy.run_trials!(launch!,()->copyto!(data,input),readback,expected,repeats;atol=1e-6,rtol=0.)
end
