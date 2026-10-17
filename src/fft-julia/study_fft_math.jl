const FFT_REVERSED=(1,5,3,7,2,6,4,8)
@inline function fft_butterfly2(d,a,b)
 x=d[a];y=d[b];Base.setindex(Base.setindex(d,x+y,a),x-y,b)
end
@inline function fft_butterfly4(d,o,inverse)
 d=fft_butterfly2(d,o+1,o+3);d=fft_butterfly2(d,o+2,o+4)
 d=Base.setindex(d,d[o+4]*ComplexF64(0,inverse ? 1 : -1),o+4)
 d=fft_butterfly2(d,o+1,o+2);fft_butterfly2(d,o+3,o+4)
end
@inline function fft_butterfly8(d,inverse)
 d=fft_butterfly2(d,1,5);d=fft_butterfly2(d,2,6);d=fft_butterfly2(d,3,7);d=fft_butterfly2(d,4,8)
 sign=inverse ? 1.0 : -1.0
 d=Base.setindex(d,(d[6]*ComplexF64(1,sign))*sqrt(0.5),6)
 d=Base.setindex(d,d[7]*ComplexF64(0,sign),7)
 d=Base.setindex(d,(d[8]*ComplexF64(-1,sign))*sqrt(0.5),8)
 d=fft_butterfly4(d,0,inverse);fft_butterfly4(d,4,inverse)
end
