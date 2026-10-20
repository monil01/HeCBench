from pathlib import Path
import json
p=Path('src/mnist-julia');ops=[]
def add(name,args,N,body):ops.append(dict(name=name,args=args,N=N,body=body))
add('apply_step_function','input,output,N','N','output[n+1]=1f0/(1f0+exp(-input[n+1]))')
add('makeError','err,output,Y,N','N','err[n+1]=(Y==n ? 1f0 : 0f0)-output[n+1]')
add('apply_grad','output,grad,N','N','output[n+1]=muladd(.1f0,grad[n+1],output[n+1])')
add('fp_preact_c1','input,preact,weight','86400','kr=n%5;kc=div(n,5)%5;ch=div(n,25)%6;r=div(n,150)%24;c=div(n,3600)%24\n  atomic_add!(preact,(ch*24+r)*24+c+1,weight[(ch*5+kr)*5+kc+1]*input[(r+kr)*28+c+kc+1])')
add('fp_bias_c1','preact,bias','3456','ch=n%6;r=div(n,6)%24;c=div(n,144)%24;i=(ch*24+r)*24+c+1\n  preact[i]+=bias[ch+1]')
add('fp_preact_s1','input,preact,weight','3456','kr=n%4;kc=div(n,4)%4;ch=div(n,16)%6;r=div(n,96)%6;c=div(n,576)%6\n  atomic_add!(preact,(ch*6+r)*6+c+1,weight[kr*4+kc+1]*input[(ch*24+r*4+kr)*24+c*4+kc+1])')
add('fp_bias_s1','preact,bias','216','ch=n%6;r=div(n,6)%6;c=div(n,36)%6;i=(ch*6+r)*6+c+1\n  preact[i]+=bias[1]')
add('fp_preact_f','input,preact,weight','2160','cl=n%10;ch=div(n,10)%6;r=div(n,60)%6;c=div(n,360)%6;i=(ch*6+r)*6+c\n  atomic_add!(preact,cl+1,weight[cl*216+i+1]*input[i+1])')
add('fp_bias_f','preact,bias','10','preact[n+1]+=bias[n+1]')
add('bp_weight_f','dweight,dpreact,poutput','2160','cl=n%10;ch=div(n,10)%6;r=div(n,60)%6;c=div(n,360)%6;i=(ch*6+r)*6+c\n  dweight[cl*216+i+1]=dpreact[cl+1]*poutput[i+1]')
add('bp_bias_f','bias,dpreact','10','bias[n+1]=muladd(.1f0,dpreact[n+1],bias[n+1])')
add('bp_output_s1','doutput,nweight,ndpreact','2160','cl=n%10;ch=div(n,10)%6;r=div(n,60)%6;c=div(n,360)%6;i=(ch*6+r)*6+c\n  atomic_add!(doutput,i+1,nweight[cl*216+i+1]*ndpreact[cl+1])')
add('bp_preact_s1','dpreact,doutput,preact','216','ch=n%6;r=div(n,6)%6;c=div(n,36)%6;i=(ch*6+r)*6+c+1;o=1f0/(1f0+exp(-preact[i]))\n  dpreact[i]=doutput[i]*o*(1f0-o)')
add('bp_weight_s1','dweight,dpreact,poutput','3456','kr=n%4;kc=div(n,4)%4;ch=div(n,16)%6;r=div(n,96)%6;c=div(n,576)%6\n  atomic_add!(dweight,kr*4+kc+1,dpreact[(ch*6+r)*6+c+1]*poutput[(ch*24+r*4+kr)*24+c*4+kc+1])')
add('bp_bias_s1','bias,dpreact','216','ch=n%6;r=div(n,6)%6;c=div(n,36)%6\n  atomic_add!(bias,1,.1f0*dpreact[(ch*6+r)*6+c+1]/216f0)')
add('bp_output_c1','doutput,nweight,ndpreact','3456','kr=n%4;kc=div(n,4)%4;ch=div(n,16)%6;r=div(n,96)%6;c=div(n,576)%6\n  atomic_add!(doutput,(ch*24+r*4+kr)*24+c*4+kc+1,nweight[kr*4+kc+1]*ndpreact[(ch*6+r)*6+c+1])')
add('bp_preact_c1','dpreact,doutput,preact','3456','ch=n%6;r=div(n,6)%24;c=div(n,144)%24;i=(ch*24+r)*24+c+1;o=1f0/(1f0+exp(-preact[i]))\n  dpreact[i]=doutput[i]*o*(1f0-o)')
add('bp_weight_c1','dweight,dpreact,poutput','86400','ch=n%6;kr=div(n,6)%5;kc=div(n,30)%5;r=div(n,150)%24;c=div(n,3600)%24\n  atomic_add!(dweight,(ch*5+kr)*5+kc+1,dpreact[(ch*24+r)*24+c+1]*poutput[(r+kr)*28+c+kc+1]/576f0)')
add('bp_bias_c1','bias,dpreact','3456','ch=n%6;r=div(n,6)%24;c=div(n,144)%24\n  atomic_add!(bias,ch+1,.1f0*dpreact[(ch*24+r)*24+c+1]/576f0)')
# Explicit Int32 divisor literals retain source32bit thread-index partition, not64bitmapping.
import re
out=['# Source-aligned source19 CNN GPU bodies; native notices preserved in original_native/.']
for op in ops:
 body=re.sub(r'(?<![\w.])(\d+)(?![\w.])',lambda m:'Int32('+m[1]+')',op['body'])
 N=op['N'] if op['N']=='N' else 'Int32('+op['N']+')'
 out.append(f"@inline function body_{op['name']}!(pos,size,{op['args']})\n Nsource={N}\n @inbounds for n=div(Nsource*pos,size):div(Nsource*(pos+Int32(1)),size)-Int32(1)\n  {body}\n end\n nothing\nend")
(p/'study_device.jl').write_text('\n'.join(out)+'\n');(p/'kernel_manifest.json').write_text(json.dumps(ops,indent=2)+'\n')
d=p/'mnist-julia-cuda.jl';d.mkdir(exist_ok=True)
out=['using CUDA','const MODEL="cuda_jl"','@inline function atomic_add!(a,i,v);CUDA.@atomic a[i]+=v;nothing;end','include("../study_device.jl")']
for op in ops:
 args=op['args'];name=op['name'];out+=[f'function kernel_{name}!({args})\n pos=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)\n size=Int32(blockDim().x)*Int32(gridDim().x)\n body_{name}!(pos,size,{args})\n nothing\nend',f'function dispatch!(::Val{{:{name}}},s,{args})\n s.launches+=1\n CUDA.@cuda threads={1 if name=="makeError" else 64} blocks={10 if name=="makeError" else 64} kernel_{name}!({args})\n nothing\nend']
out+=['include("../study_common.jl")','main()'];(d/'main.jl').write_text('\n'.join(out)+'\n')
