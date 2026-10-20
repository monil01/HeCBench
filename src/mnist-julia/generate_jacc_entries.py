#!/usr/bin/env python3
from pathlib import Path
import json
p=Path(__file__).resolve().parent;ops=json.loads((p/'kernel_manifest.json').read_text());d=p/'mnist-julia-JACC';d.mkdir(exist_ok=True)
a=['using CUDA','import JACC','JACC.@init_backend','const MODEL="jacc"','@inline function atomic_add!(a,i,v);JACC.@atomic a[i]+=v;nothing;end','include("../study_device.jl")']
for op in ops:
 name,args=op['name'],op['args'];threads=1 if name=='makeError' else 64;blocks=10 if name=='makeError' else 64;total=threads*blocks
 a+=[f'function kernel_{name}!(index,{args})\n pos=Int32(index)-Int32(1)\n body_{name}!(pos,Int32({total}),{args})\n nothing\nend',f'function dispatch!(::Val{{:{name}}},s,{args})\n s.launches+=1\n spec=JACC.launch_spec(threads={threads},blocks={blocks},shmem_size=0,sync=false)\n JACC.parallel_for(spec,{total},kernel_{name}!,{args})\n nothing\nend']
a+=['include("../study_common.jl")','main()'];(d/'main.jl').write_text('\n'.join(a)+'\n')
