function ring_guard(;initial=false)
 get(ENV,"CUDA_VISIBLE_DEVICES","")=="5,6"||error("Ring requires authorized physical visibility5,6")
 text=read(`nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits`,String)
 for line in split(strip(text),'\n')
  isempty(strip(line))&&continue
  pid=tryparse(Int,strip(line))
  pid===nothing&&error("Cannot determine GPU5 process ownership: $line")
  pid==getpid()||error("Environment blocked: external process $pid owns GPU5")
 end
 if initial
  used=parse(Int,strip(read(`nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits`,String)))
  used==0||error("Environment blocked: GPU5 memory not idle ($used MiB)")
 end
 return
end
ring_guard(initial=true)
