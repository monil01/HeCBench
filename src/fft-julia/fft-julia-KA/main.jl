using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function forward512!(work)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 gid=group*Int32(512)+tid+Int32(1);hi=tid>>3;lo=tid&Int32(7)
 smem=@localmem Float64 (576,)
 @inbounds data=(work[gid+0],work[gid+64],work[gid+128],work[gid+192],work[gid+256],work[gid+320],work[gid+384],work[gid+448])
 data=fft_butterfly8(data,false)
 for j in 2:8
  phi=(-1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/512.0)*Float64(tid)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=real(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[lo*66+hi+(i-1)*8+1]),imag(data[i])),i);end
 @synchronize
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=imag(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[lo*66+hi+(i-1)*8+1])),i);end
 @synchronize
 data=fft_butterfly8(data,false)
 for j in 2:8
  phi=(-1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/64.0)*Float64(hi)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=real(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[hi*72+lo+(i-1)*8+1]),imag(data[i])),i);end
 @synchronize
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=imag(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[hi*72+lo+(i-1)*8+1])),i);end
 data=fft_butterfly8(data,false)
 for i in 1:8;@inbounds work[gid+(i-1)*64]=false ? data[FFT_REVERSED[i]]/512.0 : data[FFT_REVERSED[i]];end
end
@kernel function inverse512!(work)
 tid=Int32(@index(Local,Linear))-Int32(1);group=Int32(@index(Group,Linear))-Int32(1)
 gid=group*Int32(512)+tid+Int32(1);hi=tid>>3;lo=tid&Int32(7)
 smem=@localmem Float64 (576,)
 @inbounds data=(work[gid+0],work[gid+64],work[gid+128],work[gid+192],work[gid+256],work[gid+320],work[gid+384],work[gid+448])
 data=fft_butterfly8(data,true)
 for j in 2:8
  phi=(1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/512.0)*Float64(tid)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=real(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[lo*66+hi+(i-1)*8+1]),imag(data[i])),i);end
 @synchronize
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=imag(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[lo*66+hi+(i-1)*8+1])),i);end
 @synchronize
 data=fft_butterfly8(data,true)
 for j in 2:8
  phi=(1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/64.0)*Float64(hi)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=real(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[hi*72+lo+(i-1)*8+1]),imag(data[i])),i);end
 @synchronize
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=imag(data[FFT_REVERSED[i]]);end
 @synchronize
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[hi*72+lo+(i-1)*8+1])),i);end
 data=fft_butterfly8(data,true)
 for i in 1:8;@inbounds work[gid+(i-1)*64]=true ? data[FFT_REVERSED[i]]/512.0 : data[FFT_REVERSED[i]];end
end
function factory(data,nfft)
 forward=forward512!(get_backend(data),64);inverse=inverse512!(get_backend(data),64)
 return (()->forward(data;ndrange=nfft*64),()->inverse(data;ndrange=nfft*64))
end
fft_study(factory)
