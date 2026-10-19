#!/usr/bin/env python3
"""Mechanical Float32/UInt32 union transcription, retaining source scalar order.
This generator does not evaluate outputs or use a numerical-library GPU SVD.
"""
from pathlib import Path
import re,json,hashlib
P=Path(__file__).resolve().parent
s=(P/'corrected_kernels.cuh').read_text();s=re.sub(r'/\*.*?\*/','',s,flags=re.S);s=re.sub(r'//[^\n]*','',s)
lines=[];stack=[]
for line in s.splitlines():
 if line.lstrip().startswith('#ifdef DEBUG_'):stack.append(False);continue
 if line.strip()=='#if 1':stack.append(True);continue
 if line.strip()=='#endif':stack.pop();continue
 if not all(stack):continue
 lines.append(line)
s='\n'.join(lines);body=s[s.index('{',s.index('void svd('))+1:s.index('__global__')];body=body[:body.rfind('}')]
unions=[]
for decl in re.findall(r'\bun\s+([^;]+);',body):unions.extend(x.strip() for x in decl.split(','))
body=re.sub(r'\bun\s+[^;]+;','',body)
body=re.sub(r'for\s*\(int i = 0; i < 8; i\+\+\)','FOR_SWEEPS',body)
constants={'gone':'UInt32(1065353216)','gsine_pi_over_eight':'UInt32(1053028117)','gcosine_pi_over_eight':'UInt32(1064076127)','gone_half':'Float32(0.5)','gsmall_number':'Float32(1e-12)','gtiny_number':'Float32(1e-20)','gfour_gamma_squared':'Float32(5.828427314758301)'}
float_pattern=r'(?<![\w.])((?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)[fF]\b'
def expr(x):
 x=re.sub(float_pattern,lambda m:'Float32('+repr(float(m.group(1)))+')',x)
 for k,v in constants.items():x=re.sub(r'\b'+k+r'\b',v,x)
 x=x.replace('fmaxf','svd_fmax')
 x=x.replace('__fadd_rn','svd_add').replace('__fsub_rn','svd_sub').replace('__frsqrt_rn','svd_rsqrt')
 x=re.sub(r'\b(\w+)\.ui\b',r'reinterpret(UInt32,\1)',x);x=re.sub(r'\b(\w+)\.f\b',r'\1',x)
 if '?' in x:
  condition,rest=x.split('?',1);yes,no=rest.split(':',1);assert yes.strip()=='0xffffffff' and no.strip()=='0';x='ifelse('+condition.strip()+',typemax(UInt32),UInt32(0))'
 return x.strip()
output=['# Generated from corrected_kernels.cuh by transcribe_source.py; source order retained.','# Native uses -use_fast_math; Julia uses ordinary typed Float32 operations.','# Rounding/contraction policy is a disclosed covariate, never a causal framework claim.','@inline svd_fmax(a::Float32,b::Float32)=ifelse(isnan(a),b,ifelse(isnan(b),a,max(a,b)))','@inline svd_add(a::Float32,b::Float32)=a+b','@inline svd_sub(a::Float32,b::Float32)=a-b','@inline svd_rsqrt(a::Float32)=1f0/sqrt(a)','@inline function source_svd('+','.join('a'+str(i)+str(j)+'::Float32' for i in range(1,4)for j in range(1,4))+')','    local '+','.join(unions)]
assigned=set();reads_before=[];statements=0
for token in re.split(r'([{};])',body):
 t=token.strip()
 if not t or t in ['{',';']:continue
 if t=='}':output.append('    end');continue
 if t=='FOR_SWEEPS':output.append('    for _sweep in 1:8');continue
 m=re.fullmatch(r'(\w+)(?:\.(f|ui))?\s*=\s*(.*)',t,re.S)
 if not m:raise ValueError('Unexpected C statement: '+t)
 name,field,rhs=m.groups()
 for var in re.findall(r'\b(\w+)\.(?:ui|f)\b',rhs):
  if var in unions and var not in assigned:reads_before.append({'read':var,'statement':t})
 assigned.add(name);rhs=expr(rhs)
 if field=='ui':rhs='reinterpret(Float32,UInt32('+rhs+'))'
 elif field=='f':rhs='Float32('+rhs+')'
 output.append('    '+name+' = '+rhs);statements+=1
assert not reads_before,reads_before
outputs=['u'+str(i)+str(j) for i in range(1,4)for j in range(1,4)]+['s11','s22','s33']+['v'+str(i)+str(j)for i in range(1,4)for j in range(1,4)]
output.extend(['    return ('+','.join(outputs)+')','end','@inline function factor_one!(input,output,n::Int32,gid::Int32)','    if gid<n','        @inbounds result=source_svd('+','.join('input[gid+Int32('+str(k)+')*n+Int32(17)]'for k in range(9))+')','        @inbounds for k in Int32(0):Int32(20)','            output[gid+k*n+Int32(17)]=result[k+Int32(1)]','        end','    end','    return nothing','end'])
p=P/'device_common.jl';p.write_text('\n'.join(output)+'\n');(P/'transcription_receipt.json').write_text(json.dumps({'source_sha256':hashlib.sha256((P/'corrected_kernels.cuh').read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scalar_assignments':statements,'union_scalars':len(unions),'linear_read_before_first_assignment':reads_before,'zero_initialization_inserted':False,'GPU_library_SVD_delegation':False,'loop_sweeps':8,'scope':'Lexical first-assignment audit plus source-order mechanical transformation; does not replace actual GPU verification.'},indent=2)+'\n');print(statements)
