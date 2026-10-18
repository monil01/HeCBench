function make_encoder(input,keys,lens,count,n,launch_flags,launch_block,launch_add,launch_scatter,launch_lengths)
 flags=CUDA.zeros(Int32,n);prefix=CUDA.zeros(Int32,n);starts=CUDA.zeros(Int32,n);levels=[];a=flags;b=prefix;size=n
 while true
  blocks=cld(size,256);totals=CUDA.zeros(Int32,blocks);push!(levels,(a,b,totals,Int32(size)))
  blocks==1&&break
  a=totals;b=CUDA.zeros(Int32,blocks);size=blocks
 end
 function encode()
  launch_flags(input,flags,Int32(n))
  for (a,b,sums,sz)in levels;launch_block(a,b,sums,sz);end
  for k in length(levels)-1:-1:1;launch_add(levels[k][2],levels[k+1][2],levels[k][4]);end
  launch_scatter(input,flags,prefix,keys,starts,Int32(n));launch_lengths(prefix,starts,lens,count,Int32(n))
 end
 encode
end
