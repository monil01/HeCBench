#!/usr/bin/env python3
from pathlib import Path
import json
p=Path(__file__).resolve().parent;ops=json.loads((p/'kernel_manifest.json').read_text());d=p/'mnist-julia-KA';d.mkdir(exist_ok=True)
a=['using CUDA,KernelAbstractions','import Atomix','const MODEL="ka"','@inline function atomic_add!(a,i,v);Atomix.@atomic :monotonic a[i]+=v;nothing;end','include("../study_device.jl")']
for op in ops:
 name,args=op['name'],op['args'];threads=1 if name=='makeError' else 64;total=10 if name=='makeError' else 4096
 a+=[f'@kernel function kernel_{name}!({args})\n pos=Int32(@index(Global,Linear))-Int32(1)\n body_{name}!(pos,Int32({total}),{args})\nend',f'function dispatch!(::Val{{:{name}}},s,{args})\n s.launches+=1\n kernel_{name}!(CUDABackend(),{threads})({args};ndrange={total})\n nothing\nend']
a+=['include("../study_common.jl")','main()'];(d/'main.jl').write_text('\n'.join(a)+'\n')
