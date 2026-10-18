function study_guard(;initial=false)
 visibility=get(ENV,"CUDA_VISIBLE_DEVICES","")
 visibility in ("6","5,6")||error("OnlyauthorizedGPU6or5,6visibility")
 if visibility=="5,6"
  text=read(`nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits`,String)
  for line in split(strip(text),'\n')
   isempty(strip(line))&&continue
   pid=tryparse(Int,strip(line));pid===nothing&&error("Cannot determineGPU5owner")
   pid==getpid()||error("ExternalGPU5PID $pid; defer")
  end
  if initial
   used=parse(Int,strip(read(`nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits`,String)))
   used==0||error("GPU5notidle ($used MiB); defer")
  end
 end
 nothing
end
study_guard(initial=true)
