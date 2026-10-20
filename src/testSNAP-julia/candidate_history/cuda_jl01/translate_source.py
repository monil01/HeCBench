#!/usr/bin/env python3
"""Source-bound mechanical C AST emitter, not an independent scientific oracle.
Requires pycparser; generated Julia is reviewed and compiled separately.
"""
from pathlib import Path
import sys,re,subprocess,json,hashlib
from pycparser import c_parser,c_ast as A
p=Path(__file__).resolve().parent
main=(p/'original_native/main.cu').read_text();utils=(p/'original_native/utils.cu').read_text();snap=(p/'original_native/snap.h').read_text()
mac='\n'.join(x for x in snap.splitlines() if x.startswith('#define') and any(y in x for y in ['INDEX','MY_PI','rfac0','rmin0']))
# Join source macro continuations before selecting definitions.
snap=snap.replace('\\\n','');mac='\n'.join(x for x in snap.splitlines() if x.startswith('#define') and any(y in x for y in ['INDEX','MY_PI','rfac0','rmin0']))
src=utils[utils.index('__device__'):utils.index('\nvoid options') if '\nvoid options' in utils else utils.index('double elapsedTime')]
# Only three device definitions; avoid intervening host utility declarations.
start=utils.index('__device__');end=utils.index('\n}',utils.index('void compute_duarray'))+2
src=utils[start:end]+'\n'+main[main.index('__global__'):main.index('\nint main(')]
src=re.sub(r'/\*.*?\*/|//[^\n]*','',src,flags=re.S);src=src.replace('__device__','').replace('__global__','').replace('__restrict__','');src=re.sub(r'=\s*\{', '= (COMPLEX){',src)
pre=mac+'\ntypedef struct { double re; double im; } COMPLEX;\n'+src
clean=subprocess.run(['cpp','-P','-'],input=pre,text=True,capture_output=True,check=True).stdout
ast=c_parser.CParser().parse(clean)
class Emit:
 def __init__(self):self.scopes=[{}];self.types={};self.count=0;self.lines=[];self.level=1
 def put(self,s):self.lines.append('    '*self.level+s)
 def name(self,n):
  for d in reversed(self.scopes):
   if n in d:return d[n]
  return n
 def base(self,t):
  while hasattr(t,'type'):t=t.type
  return ' '.join(t.names) if isinstance(t,A.IdentifierType) else 'COMPLEX'
 def isint(self,n):
  if isinstance(n,A.Constant):return n.type=='int'
  if isinstance(n,A.ID):return self.types.get(self.name(n.name))=='int'
  if isinstance(n,A.ArrayRef):return self.types.get(self.name(n.name.name))=='int'
  if isinstance(n,A.BinaryOp):return self.isint(n.left) and self.isint(n.right)
  if isinstance(n,A.Cast):return self.base(n.to_type)=='int'
  return False
 def ex(self,n):
  if isinstance(n,A.ID):return self.name(n.name)
  if isinstance(n,A.Constant):return 'Int32('+n.value+')' if n.type=='int' else n.value
  if isinstance(n,A.ArrayRef):return self.ex(n.name)+'[('+self.ex(n.subscript)+')+Int32(1)]'
  if isinstance(n,A.StructRef):
   assert isinstance(n.name,A.ArrayRef)
   return self.ex(n.name.name)+'[Int32(2)*('+self.ex(n.name.subscript)+')+Int32('+('1' if n.field.name=='re' else '2')+')]'
  if isinstance(n,A.BinaryOp):
   a,b=self.ex(n.left),self.ex(n.right)
   if n.op=='/' and self.isint(n.left) and self.isint(n.right):return 'div('+a+','+b+')'
   return '('+a+' '+n.op+' '+b+')'
  if isinstance(n,A.UnaryOp):
   if n.op in ('+','-','!'):return '('+n.op+self.ex(n.expr)+')'
   if n.op=='&':return self.ex(n.expr)
   raise ValueError(n.op)
  if isinstance(n,A.FuncCall):
   name=n.name.name;args=n.args.exprs if n.args else []
   if name=='atomicAdd':
    t=args[0].expr;assert isinstance(t,A.StructRef);return 'snap_atomic!('+self.ex(t.name.name)+',Int32(2)*('+self.ex(t.name.subscript)+')+Int32('+('1' if t.field.name=='re' else '2')+'),'+self.ex(args[1])+')'
   if name=='pow':return '('+self.ex(args[0])+' ^ '+self.ex(args[1])+')'
   return name+'('+','.join(self.ex(x) for x in args)+')'
  if isinstance(n,A.Cast):return ('Int32' if self.base(n.to_type)=='int' else 'Float64')+'('+self.ex(n.expr)+')'
  raise ValueError(type(n).__name__)
 def decl(self,n):
  self.count+=1;v=n.name+'_s'+str(self.count);self.scopes[-1][n.name]=v;self.types[v]=self.base(n.type)
  if isinstance(n.type,A.ArrayDecl):self.put(v+' = MVector{'+n.type.dim.value+',Float64}(undef)')
  elif n.init:
   if isinstance(n.type,A.PtrDecl) and isinstance(n.init,A.BinaryOp) and n.init.op=='+':self.put(v+' = SnapShift('+self.ex(n.init.left)+','+self.ex(n.init.right)+')')
   else:self.put(v+' = '+self.ex(n.init))
  else:self.put('local '+v)
 def block(self,n):
  self.scopes.append({})
  for x in (n.block_items or []):self.st(x)
  self.scopes.pop()
 def st(self,n):
  if isinstance(n,A.Compound):self.block(n)
  elif isinstance(n,A.Decl):self.decl(n)
  elif isinstance(n,A.DeclList):
   for d in n.decls:self.decl(d)
  elif isinstance(n,A.Assignment):
   if isinstance(n.rvalue,A.CompoundLiteral):
    t=n.lvalue;assert isinstance(t,A.ArrayRef)
    for field,val in zip(['re','im'],n.rvalue.init.exprs):self.put(self.ex(A.StructRef(t,'.',A.ID(field)))+' = '+self.ex(val))
   else:self.put(self.ex(n.lvalue)+' '+n.op+' '+self.ex(n.rvalue))
  elif isinstance(n,A.If):
   self.put('if '+self.ex(n.cond));self.level+=1;self.st(n.iftrue);self.level-=1
   if n.iffalse:self.put('else');self.level+=1;self.st(n.iffalse);self.level-=1
   self.put('end')
  elif isinstance(n,A.For):
   self.scopes.append({});self.st(n.init);self.put('while '+self.ex(n.cond));self.level+=1;self.st(n.stmt);self.st(n.next);self.level-=1;self.put('end');self.scopes.pop()
  elif isinstance(n,A.UnaryOp) and n.op in ('p++','++','p--','--'):self.put(self.ex(n.expr)+(' += Int32(1)' if '+' in n.op else ' -= Int32(1)'))
  elif isinstance(n,A.FuncCall):self.put(self.ex(n))
  elif isinstance(n,A.Return):self.put('return '+(self.ex(n.expr) if n.expr else 'nothing'))
  else:raise ValueError(type(n).__name__)
functions=[];manifest=[]
for f in ast.ext:
 if not isinstance(f,A.FuncDef):continue
 e=Emit();name=f.decl.name;params=f.decl.type.args.params;names=[x.name for x in params]
 for x in params:e.types[x.name]=e.base(x.type)
 iskernel=name not in ('compute_sfac','compute_dsfac','compute_duarray')
 if iskernel:
  # Replace CUDA launch coordinates with explicit zero-based per-thread inputs.
  class Replace(A.NodeVisitor):
   def visit_Decl(self,n):
    if n.init and 'blockIdx' in str(n.init):n.init=A.ID('snap_ix' if 'x' in str(n.init) else 'snap_iy')
  # Exact AST detect first thread-derived declarations by their source names.
  for st in f.body.block_items or []:
   if isinstance(st,A.Decl) and st.name in ('i','natom','nbor','jjz') and st.init and isinstance(st.init,A.BinaryOp) and isinstance(st.init.right,A.StructRef):st.init=A.ID('snap_iy' if st.name in ('nbor','jjz') else 'snap_ix')
  names+=['snap_ix','snap_iy'];e.types.update(snap_ix='int',snap_iy='int')
 e.block(f.body);functions.append(('@inline ' if not iskernel else '')+'function '+name+'('+','.join(names)+')\n    @inbounds begin\n'+'\n'.join(e.lines)+'\n    end\n    return nothing\nend\n')
 if not iskernel and name!='compute_duarray':functions[-1]=functions[-1].replace('    return nothing\nend','    return 0.0\nend')
 manifest.append({'name':name,'parameters':names,'kind':'kernel_body' if iskernel else 'device_helper'})
preamble='''# Derived from Sandia TestSNAP Zero-Clause BSD source. See original_native/ attribution.
using StaticArrays
struct SnapShift{A}
    array::A
    offset::Int32
end
@inline Base.getindex(s::SnapShift,i::Integer)=s.array[s.offset+i]
'''
(p/'study_device.jl').write_text(preamble+'\n'.join(functions));(p/'source_translation_manifest.json').write_text(json.dumps({'method':'C AST lexical scope/int32 arithmetic/one-based arrays; COMPLEX flattened AoS Float64 re/im; exact seven source kernels, three helpers; source expressions preserved','functions':manifest,'source_sha256':{f:hashlib.sha256((p/'original_native'/f).read_bytes()).hexdigest() for f in ['main.cu','utils.cu','snap.h']}},indent=2)+'\n');print('emitted',len(functions),'functions')
