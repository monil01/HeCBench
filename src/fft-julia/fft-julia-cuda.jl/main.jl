using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function forward512!(work)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 gid=group*Int32(512)+tid+Int32(1);hi=tid>>3;lo=tid&Int32(7)
 smem=CuStaticSharedArray(Float64,576)
 data=ntuple(i->(@inbounds work[gid+(i-1)*64]),Val(8))
 data=fft_butterfly8(data,false)
 for j in 2:8
  phi=(-1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/512.0)*Float64(tid)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=real(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[lo*66+hi+(i-1)*8+1]),imag(data[i])),i);end
 sync_threads()
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=imag(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[lo*66+hi+(i-1)*8+1])),i);end
 sync_threads()
 data=fft_butterfly8(data,false)
 for j in 2:8
  phi=(-1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/64.0)*Float64(hi)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=real(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[hi*72+lo+(i-1)*8+1]),imag(data[i])),i);end
 sync_threads()
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=imag(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[hi*72+lo+(i-1)*8+1])),i);end
 data=fft_butterfly8(data,false)
 for i in 1:8;@inbounds work[gid+(i-1)*64]=false ? data[FFT_REVERSED[i]]/512.0 : data[FFT_REVERSED[i]];end
end
function inverse512!(work)
 tid=Int32(threadIdx().x)-Int32(1);group=Int32(blockIdx().x)-Int32(1)
 gid=group*Int32(512)+tid+Int32(1);hi=tid>>3;lo=tid&Int32(7)
 smem=CuStaticSharedArray(Float64,576)
 data=ntuple(i->(@inbounds work[gid+(i-1)*64]),Val(8))
 data=fft_butterfly8(data,true)
 for j in 2:8
  phi=(1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/512.0)*Float64(tid)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=real(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[lo*66+hi+(i-1)*8+1]),imag(data[i])),i);end
 sync_threads()
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*66+1]=imag(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[lo*66+hi+(i-1)*8+1])),i);end
 sync_threads()
 data=fft_butterfly8(data,true)
 for j in 2:8
  phi=(1.0*2.0*pi*Float64(FFT_REVERSED[j]-1)/64.0)*Float64(hi)
  data=Base.setindex(data,data[j]*ComplexF64(cos(phi),sin(phi)),j)
 end
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=real(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(@inbounds(smem[hi*72+lo+(i-1)*8+1]),imag(data[i])),i);end
 sync_threads()
 for i in 1:8;@inbounds smem[hi*8+lo+(i-1)*72+1]=imag(data[FFT_REVERSED[i]]);end
 sync_threads()
 for i in 1:8;data=Base.setindex(data,ComplexF64(real(data[i]),@inbounds(smem[hi*72+lo+(i-1)*8+1])),i);end
 data=fft_butterfly8(data,true)
 for i in 1:8;@inbounds work[gid+(i-1)*64]=true ? data[FFT_REVERSED[i]]/512.0 : data[FFT_REVERSED[i]];end
end
function factory(data,nfft)
 return (()->(@cuda threads=64 blocks=nfft forward512!(data)),()->(@cuda threads=64 blocks=nfft inverse512!(data)))
end
if abspath(PROGRAM_FILE)==@__FILE__;fft_study(factory);end
