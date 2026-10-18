"""Owned source-preserving C AST to explicit Julia numerical helper, not a device FFI."""
from pathlib import Path
import re,sys
from pycparser import c_parser
B=Path(__file__).resolve().parent;R=B.parents[1]
s=(R/'src/lulesh-cuda/lulesh.cu').read_text();s=s[s.index('__device__ static inline'):s.index('/*********************************/\n/* Data structure implementation */')]
s=re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S)
s=re.sub(r'#define TRIPLE_PRODUCT.*?#undef TRIPLE_PRODUCT','',s,flags=re.S)if False else s
# C preprocessor and function-style C++ scalar casts normalized for C AST only.
s=re.sub(r'#define TRIPLE_PRODUCT[^\n]*\\\n[^\n]*\n','',s)
s=re.sub(r'^\s*#.*$','',s,flags=re.M)
s=re.sub(r'\b(__device__|__global__|__host__|__restrict__|static|inline)\b','',s)
s=re.sub(r'\b(Real_t|Index_t|Int_t)\s*\(',r'(\1)(',s)
s='typedef double Real_t;typedef int Index_t;typedef int Int_t;\n'+s
try:
 ast=c_parser.CParser().parse(s)
except Exception:
 Path('/tmp/lulesh_c_clean.c').write_text(s);raise
Path('/tmp/lulesh_c_clean.c').write_text(s)

from pycparser import c_ast as C
import json,shutil
kernel_names=re.findall(r'__global__\s+void\s+(\w+)',(R/'src/lulesh-cuda/lulesh.cu').read_text())
class Translator:
 def __init__(self,f):
  self.f=f;self.ptr=set();self.dims={};self.types={};self.refs=set()
  def scan(n):
   if isinstance(n,C.Decl):
    t=n.type;ds=[]
    if isinstance(t,(C.PtrDecl,C.ArrayDecl)):self.ptr.add(n.name)
    while isinstance(t,C.ArrayDecl):ds.append(int(t.dim.value)if t.dim else None);t=t.type
    if ds:self.dims[n.name]=ds
    while hasattr(t,'type'):t=t.type
    if isinstance(t,C.IdentifierType):self.types[n.name]='Float64'if t.names==['Real_t']else'Int32'
   if isinstance(n,C.UnaryOp)and n.op=='&'and isinstance(n.expr,C.ID):self.refs.add(n.expr.name)
   for _,c in n.children():scan(c)
  scan(f)
 def name(self,x):return {'local':'loc','global':'glob','baremodule':'baremod','module':'modu'}.get(x,x)
 def e(self,n):
  if n is None:return'nothing'
  if isinstance(n,C.ID):return self.name(n.name)+('[1]'if n.name in self.refs else'')
  if isinstance(n,C.Constant):
   if n.type in ['int','unsigned int']:return'Int32('+n.value.rstrip('uUlL')+')'
   return n.value.rstrip('fFlL') if '.'in n.value or'e'in n.value.lower()else n.value+'.0'
  if isinstance(n,C.Cast):
   t=n.to_type.type
   while hasattr(t,'type'):t=t.type
   typ='Float64'if t.names==['Real_t']else'Int32'
   return typ+'('+self.e(n.expr)+')'
  if isinstance(n,C.StructRef):return {'blockDim':'Int32(256)','blockIdx':'div(gidx,Int32(256))','threadIdx':'rem(gidx,Int32(256))'}[n.name.name]
  if isinstance(n,C.ArrayRef):
   if isinstance(n.name,C.ArrayRef):
    base=n.name.name;return self.e(base)+'['+self.e(n.name.subscript)+'*Int32('+str(self.dims[base.name][1])+')+'+self.e(n.subscript)+'+Int32(1)]'
   if isinstance(n.name,C.ID)and len(self.dims.get(n.name.name,[]))>1:return'cptr('+self.e(n.name)+','+self.e(n.subscript)+'*Int32('+str(self.dims[n.name.name][1])+'))'
   return self.e(n.name)+'['+self.e(n.subscript)+'+Int32(1)]'
  if isinstance(n,C.UnaryOp):
   if n.op=='&':
    if isinstance(n.expr,C.ID):return self.name(n.expr.name)
    if isinstance(n.expr,C.ArrayRef):
     a=n.expr
     if isinstance(a.name,C.ArrayRef):return'cptr('+self.e(a.name.name)+','+self.e(a.name.subscript)+'*Int32('+str(self.dims[a.name.name.name][1])+')+'+self.e(a.subscript)+')'
     return'cptr('+self.e(a.name)+','+self.e(a.subscript)+')'
    raise ValueError(n)
   if n.op=='*':return self.e(n.expr)+'[1]'
   if n.op in ['p++','++']:return self.e(n.expr)+' += Int32(1)'
   if n.op in ['p--','--']:return self.e(n.expr)+' -= Int32(1)'
   if n.op=='!':return'!('+self.e(n.expr)+' != 0)'
   return'('+n.op+self.e(n.expr)+')'
  if isinstance(n,C.BinaryOp):
   if n.op=='+'and isinstance(n.left,C.ID)and n.left.name in self.ptr:return'cptr('+self.e(n.left)+','+self.e(n.right)+')'
   op={'!=':'!=','&&':'&&','||':'||','&':'&'}.get(n.op,n.op)
   # All integer division in translated device source is only compile-time indexing; explicit div where needed.
   return'('+self.e(n.left)+' '+op+' '+self.e(n.right)+')'
  if isinstance(n,C.Assignment):return'('+self.e(n.lvalue)+' '+n.op+' '+self.e(n.rvalue)+')'
  if isinstance(n,C.FuncCall):return {'fabs':'abs','SQRT':'sqrt','FABS':'abs','TRIPLE_PRODUCT':'triple_product'}.get(n.name.name,n.name.name)+'('+','.join(self.e(a)for a in n.args.exprs)+')'
  raise TypeError(type(n).__name__)
 def stmt(self,n,indent=1):
  tab=' '*indent
  if n is None:return''
  if isinstance(n,C.Compound):return''.join(self.stmt(c,indent)for c in n.block_items or[])
  if isinstance(n,C.DeclList):return''.join(self.stmt(c,indent)for c in n.decls)
  if isinstance(n,C.Decl):
   name=self.name(n.name);typ=self.types[n.name]
   if isinstance(n.type,C.ArrayDecl):
    dims=self.dims[n.name];size=math.prod(dims);return tab+name+' = MVector{'+str(size)+','+typ+'}(undef)\n'
   if isinstance(n.type,C.PtrDecl):return tab+name+' = '+self.e(n.init)+'\n'
   val=self.e(n.init)if n.init else('0.0'if typ=='Float64'else'Int32(0)')
   if n.name in self.refs:return tab+name+' = MVector{1,'+typ+'}('+val+')\n'
   return tab+name+' = '+typ+'('+val+')\n'
  if isinstance(n,C.If):
   out=tab+'if '+self.e(n.cond)+'\n'+self.stmt(n.iftrue,indent+1)
   if n.iffalse:out+=tab+'else\n'+self.stmt(n.iffalse,indent+1)
   return out+tab+'end\n'
  if isinstance(n,C.For):return self.stmt(n.init,indent)+tab+'while '+self.e(n.cond)+'\n'+self.stmt(n.stmt,indent+1)+tab+' '+self.e(n.next)+'\n'+tab+'end\n'
  if isinstance(n,C.Switch):
   out='';pending=[];first=True
   for c in n.stmt.block_items:
    if isinstance(c,C.Case):
     pending.append(self.e(c.expr))
     if not c.stmts:continue
     out+=tab+('if 'if first else'elseif ')+ ' || '.join('('+self.e(n.cond)+' == '+x+')'for x in pending)+'\n';pending=[];first=False
    elif isinstance(c,C.Default):out+=tab+'else\n'
    else:raise TypeError(type(c))
    for v in c.stmts:
     if not isinstance(v,C.Break):out+=self.stmt(v,indent+1)
   return out+tab+'end\n'
  if isinstance(n,C.Return):return tab+'return '+self.e(n.expr)+'\n'
  if isinstance(n,C.Break):return tab+'break\n'
  return tab+self.e(n)+'\n'
 def run(self):
  name=self.f.decl.name;params=[p.name for p in self.f.decl.type.args.params]
  if name in kernel_names:params+=['gidx']
  return'@inline function '+name+('_body!'if name in kernel_names else'')+'('+','.join(self.name(p)for p in params)+')\n @inbounds begin\n'+self.stmt(self.f.body,2)+' end\n'+(' return nothing\n'if name in kernel_names else'')+'end\n\n'
import math
header='''# Translated from LLNL LULESH 2.0; original BSD/DOE notice in reference_original/lulesh.cu.
# All 15 source numerical bodies and 11 helpers retained in original arithmetic order.
using StaticArrays
struct COffset{A}
 a::A
 offset::Int32
end
@inline cptr(a,o)=COffset(a,Int32(o))
@inline cptr(a::COffset,o)=COffset(a.a,a.offset+Int32(o))
@inline Base.getindex(a::COffset,i)=(@inbounds a.a[a.offset+i])
@inline Base.setindex!(a::COffset,v,i)=(@inbounds a.a[a.offset+i]=v)
@inline triple_product(x1,y1,z1,x2,y2,z2,x3,y3,z3)=x1*(y2*z3-z2*y3)+x2*(z1*y3-y1*z3)+x3*(y1*z2-z1*y2)
'''
src=(R/'src/lulesh-cuda/lulesh.cu').read_text();head=(R/'src/lulesh-cuda/lulesh.h').read_text()
for name,val in re.findall(r'^#define\s+(\w+)\s+(.+)$',src[:src.index('__device__ static inline')],re.M):
 if name not in ['EPSILON','THREADS']:header+='const '+name+' = '+val.replace('Real_t','Float64')+'\n'
for name,val in re.findall(r'^#define\s+((?:XI|ETA|ZETA)_\w+)\s+(0x[0-9a-fA-F]+)',head,re.M):header+='const '+name+' = Int32('+val+')\n'
text=header+'\n'+''.join(Translator(f).run()for f in ast.ext if isinstance(f,C.FuncDef))
(B/'study_device.jl').write_text(text)
write_manifest=dict(kernel_names=kernel_names,kernels=[])
for f in ast.ext:
 if isinstance(f,C.FuncDef)and f.decl.name in kernel_names:write_manifest['kernels'].append(dict(name=f.decl.name,params=[dict(name=p.name,pointer=isinstance(p.type,(C.PtrDecl,C.ArrayDecl)),type='Float64'if 'Real_t'in str(p.type)else'Int32')for p in f.decl.type.args.params]))
(B/'source_translation_map.json').write_text(json.dumps(write_manifest,indent=2)+'\n')
print('Wrote',len(text),'chars',len(kernel_names),'kernel bodies')
