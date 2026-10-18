"""Generate explicit portable lane operations from the audited native operation inventory."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;specs=json.loads((p/'operation_specs.json').read_text())['operations'];out=['# Authored portable lane expansion of all118 source operations; no vendor calls.\n']
for s in specs:
 idx,w,sgn,op,rel=s['id'],s['bits'],s['signed'],s['op'],s['relu'];mask=(1<<w)-1;half=1<<(w-1);lines=[f'@inline function simd_op_{idx}(a::UInt32,b::UInt32,c::UInt32)']
 if w==32:
  if sgn:lines+=[' x=reinterpret(Int32,a);y=reinterpret(Int32,b);z=reinterpret(Int32,c)']
  else:lines+=[' x=a;y=b;z=c']
  add='reinterpret(Int32,a+b)'if sgn else'a+b';ex={'addmax':f'max({add},z)','addmin':f'min({add},z)','max':'max(x,y)','min':'min(x,y)','max3':'max(max(x,y),z)','min3':'min(min(x,y),z)'}[op];lines+=[' t='+ex]
  if rel:lines+=[' t=max(t,Int32(0))']
  lines+=[' return '+('reinterpret(UInt32,t)'if sgn else't'),'end\n'];out.append('\n'.join(lines));continue
 lines+=[f' mask=UInt32({mask});result=UInt32(0)',f' for lane in Int32(0):Int32({32//w-1})',f'  shift=lane*Int32({w});x=reinterpret(Int32,(a>>shift)&mask);y=reinterpret(Int32,(b>>shift)&mask);z=reinterpret(Int32,(c>>shift)&mask)']
 if sgn:lines +=[f'  x=x>=Int32({half}) ? x-Int32({1<<w}) : x;y=y>=Int32({half}) ? y-Int32({1<<w}) : y;z=z>=Int32({half}) ? z-Int32({1<<w}) : z']
 lo=-(1<<(w-1))if sgn else 0;hi=(1<<(w-1))-1 if sgn else mask
 if op in ['addmax','addmin']:
  lines +=[f'  u=(x+y)&Int32({mask})']
  if sgn:lines +=[f'  u=u>=Int32({half}) ? u-Int32({1<<w}) : u']
  ex=('max'if op=='addmax'else'min')+'(u,z)'
 elif op.startswith(('cmp_','set_')):
  kind,comp=op.split('_');cond={'eq':'x==y','ne':'x!=y','ge':'x>=y','gt':'x>y','le':'x<=y','lt':'x<y'}[comp];ex=f'({cond}) ? Int32({mask if kind=="cmp"else 1}) : Int32(0)'
 else:
  ex={'abs':'abs(x)','absdiff':'abs(x-y)','abssat':f'min(abs(x),Int32({hi}))','add':'x+y','sub':'x-y','addsat':f'clamp(x+y,Int32({lo}),Int32({hi}))','subsat':f'clamp(x-y,Int32({lo}),Int32({hi}))','avg':'((x+y)>=0 ? (x+y+Int32(1)) : (x+y))>>Int32(1)','hadd':'(x+y)>>Int32(1)','max':'max(x,y)','min':'min(x,y)','neg':'-x','negsat':f'min(-x,Int32({hi}))','sad':'abs(x-y)','max3':'max(max(x,y),z)','min3':'min(min(x,y),z)'}[op]
 lines+=['  t='+ex]
 if rel:lines+=['  t=max(t,Int32(0))']
 if op=='sad':lines+=['  result+=reinterpret(UInt32,t)']
 else:lines+=['  result|=(reinterpret(UInt32,t)&mask)<<shift']
 lines+=[' end',' return result','end\n'];out.append('\n'.join(lines))
out+=['@inline function simd_xor(a::UInt32,b::UInt32,c::UInt32)',' r=simd_op_0(a,b,c)']+[' r⊻=simd_op_'+str(i)+'(a,b,c)'for i in range(1,118)]+[' return r','end\n','@inline function individual_operation(op::Int32,a::UInt32,b::UInt32,c::UInt32)']
for i in range(118):out+=[(' if'if i==0 else' elseif')+f' op==Int32({i})',f'  return simd_op_{i}(a,b,c)']
out+=[' end',' return UInt32(0) # unreachable: everydeclared probe op is0..117','end\n'];(p/'study_operations.jl').write_text('\n'.join(out))
