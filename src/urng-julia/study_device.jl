@inline function urng_step(seed::Int32)
 k=seed÷Int32(127773);seed=Int32(16807)*(seed-k*Int32(127773))-Int32(2836)*k
 return seed<0 ? seed+Int32(2147483647) : seed
end
@inline function urng_pixel(pixel,lane,shuffle)
 x=Float32(pixel[1]);y=Float32(pixel[2]);z=Float32(pixel[3]);w=Float32(pixel[4]);avg=(x+y+z+w)/4f0;seed=-trunc(Int32,avg)
 for j in Int32(16):-Int32(1):Int32(0)
  seed=urng_step(seed)
  if j<Int32(16);@inbounds shuffle[Int32(256)*j+lane]=seed;end
 end
 @inbounds iy=shuffle[lane];seed=urng_step(seed);j=iy÷Int32(134217728);iy=shuffle[Int32(256)*j+lane]
 u=(1f0/Float32(2147483647))*Float32(iy);dev=(u-.55f0)*25f0
 return (urng_clamp(x+dev),urng_clamp(y+dev),urng_clamp(z+dev),urng_clamp(w+dev))
end
@inline urng_clamp(x)=UInt8(trunc(Int32,clamp(x,0f0,255f0)))
