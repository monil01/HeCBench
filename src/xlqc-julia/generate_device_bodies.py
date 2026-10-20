#!/usr/bin/env python3
"""Translate the original Rys C math with explicit C storage casts and row-major locals."""
from pathlib import Path
import argparse,re,subprocess,json,hashlib
from pycparser import c_parser,c_ast as A
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);z=a.parse_args();pieces=[]
for name in ['cuda_rys_sp.cu','cuda_rys_dp.cu']:
 s=(P/'reference_history'/name).read_text();s=s[s.index('__device__'):];s=re.sub(r'^\s*#include.*$','',s,flags=re.M);s=s.replace('__device__','').replace('__global__','').replace('__shared__','').replace('__restrict','');pieces.append(s)
preamble='''typedef struct {int x,y,z;} dim3; dim3 threadIdx,blockIdx;
#define MAXROOTS 7
#define BLOCKSIZE 8
#define SCREEN_THR 1.0e-16
#define M_PI 3.14159265358979323846
'''
raw=subprocess.run(['gcc','-E','-P','-x','c','-'],input=preamble+'\n'.join(pieces),text=True,capture_output=True,check=True).stdout;ast=c_parser.CParser().parse(raw)
def type_info(t):
 dims=[];pointer=False
 while isinstance(t,(A.PtrDecl,A.ArrayDecl)):
  if isinstance(t,A.PtrDecl):pointer=True
  else:dims.append(t.dim)
  t=t.type
 assert isinstance(t,A.TypeDecl),type(t)
 names=t.type.names;typ='Float32'if names==['float']else'Float64'if names==['double']else'Nothing'if names==['void']else'Int32'
 return typ,pointer,dims
functions=[x for x in ast.ext if isinstance(x,A.FuncDef)];return_types={f.decl.name:type_info(f.decl.type.type)[0]for f in functions};return_types.update(sqrtf='Float32',expf='Float32',powf='Float32',sqrt='Float64',exp='Float64',pow='Float64',fabs='Float64')
class Generator:
 def __init__(self,f):
  self.f=f;self.types={};self.ptrs=set();self.ranks={};self.ret=return_types[f.decl.name]
  class Visit(A.NodeVisitor):
   def visit_Decl(v,n):
    if not isinstance(n.type,A.FuncDecl):
     typ,ptr,dims=type_info(n.type);self.types[n.name]=typ
     if ptr:self.ptrs.add(n.name)
     if dims:self.ranks[n.name]=len(dims)
    v.generic_visit(n)
  Visit().visit(f)
 def typ(self,n):
  if isinstance(n,A.ID):return self.types.get(n.name,'Int32')
  if isinstance(n,A.Constant):return'Float32'if n.type=='float'else'Float64'if n.type=='double'else'Int32'
  if isinstance(n,A.ArrayRef):return self.typ(n.name)
  if isinstance(n,A.Cast):return type_info(n.to_type.type)[0]
  if isinstance(n,A.FuncCall):return return_types.get(n.name.name,'Int32')
  if isinstance(n,A.UnaryOp):return self.typ(n.expr)
  if isinstance(n,A.TernaryOp):types=[self.typ(n.iftrue),self.typ(n.iffalse)]
  elif isinstance(n,A.BinaryOp):
   if n.op in ['<','>','<=','>=','==','!=','&&','||']:return'Int32'
   types=[self.typ(n.left),self.typ(n.right)]
  else:return'Int32'
  return'Float64'if'Float64'in types else'Float32'if'Float32'in types else'Int32'
 def cast(self,lhs,rhs):
  if isinstance(lhs,A.ID)and lhs.name not in self.ptrs:return self.types[lhs.name]+'('+rhs+')'
  return rhs
 def exp(self,n):
  if isinstance(n,A.ID):return n.name
  if isinstance(n,A.Constant):
   if n.type=='string':return n.value
   return self.typ(n)+'('+n.value.rstrip('fFlLuU')+')'
  if isinstance(n,A.BinaryOp):
   x,y=self.exp(n.left),self.exp(n.right)
   if n.op in ['&&','||']:return f'(xl_truth({x}) {n.op} xl_truth({y}))'
   if n.op=='/'and self.typ(n)=='Int32':return f'div({x},{y})'
   if n.op=='%':return f'rem({x},{y})'
   return f'({x} {n.op} {y})'
  if isinstance(n,A.UnaryOp):
   x=self.exp(n.expr)
   if n.op=='&':assert isinstance(n.expr,A.ArrayRef);return f'XLPtr({self.exp(n.expr.name)},{self.exp(n.expr.subscript)})'
   if n.op=='*':return f'{x}[Int32(1)]'
   if n.op=='!':return f'(!xl_truth({x}))'
   if n.op in ['p++','++']:return f'({x} += Int32(1))'
   if n.op in ['p--','--']:return f'({x} -= Int32(1))'
   return f'({n.op}{x})'
  if isinstance(n,A.ArrayRef):
   if isinstance(n.name,A.ArrayRef):return f'{self.exp(n.name.name)}[{self.exp(n.name.subscript)}*Int32(7)+{self.exp(n.subscript)}+Int32(1)]'
   return f'{self.exp(n.name)}[{self.exp(n.subscript)}+Int32(1)]'
  if isinstance(n,A.StructRef):return f'(Int32(CUDA.{n.name.name}().{n.field.name})-Int32(1))'
  if isinstance(n,A.TernaryOp):return f'(xl_truth({self.exp(n.cond)}) ? {self.exp(n.iftrue)} : {self.exp(n.iffalse)})'
  if isinstance(n,A.Assignment):
   lhs=self.exp(n.lvalue);rhs=self.exp(n.rvalue)
   if n.op!='=':rhs=f'({lhs} {n.op[:-1]} {rhs})'
   return f'({lhs} = {self.cast(n.lvalue,rhs)})'
  if isinstance(n,A.FuncCall):
   name=n.name.name;args=n.args.exprs if n.args else[]
   if name=='__syncthreads':return'CUDA.sync_threads()'
   if name=='printf':return'CUDA.@cuprintf('+','.join(self.exp(x)for x in args)+')'
   name={'sqrtf':'sqrt','sqrt':'sqrt','expf':'exp','powf':'xl_pow','pow':'xl_pow','fabs':'abs'}.get(name,name)
   return name+'('+','.join(self.exp(x)for x in args)+')'
  if isinstance(n,A.Cast):return type_info(n.to_type.type)[0]+'('+self.exp(n.expr)+')'
  if isinstance(n,A.ExprList):return'(begin '+'; '.join(self.exp(x)for x in n.exprs)+' end)'
  raise TypeError(('expression',type(n),n))
 def stmt(self,n,level=1):
  pad='    '*level
  if n is None:return''
  if isinstance(n,A.Compound):return'\n'.join(self.stmt(x,level)for x in n.block_items or[])
  if isinstance(n,A.DeclList):return'\n'.join(self.stmt(x,level)for x in n.decls)
  if isinstance(n,A.Decl):
   typ,ptr,dims=type_info(n.type)
   if dims:
    count=1
    def dimval(d):
     if isinstance(d,A.Constant):return int(d.value)
     assert isinstance(d,A.BinaryOp)and d.op=='*';return dimval(d.left)*dimval(d.right)
    for d in dims:count*=dimval(d)
    alloc=f'CUDA.CuStaticSharedArray({typ},{count})'if n.name.startswith('elem_')else f'MVector{{{count},{typ}}}(undef)'
    assert n.init is None;return pad+n.name+' = '+alloc
   if n.init is None:return pad+'local '+n.name+('::'+typ if not ptr else'')
   return pad+n.name+' = '+(self.exp(n.init)if ptr else typ+'('+self.exp(n.init)+')')
  if isinstance(n,A.If):
   s=pad+'if xl_truth('+self.exp(n.cond)+')\n'+self.stmt(n.iftrue,level+1)
   if n.iffalse:s+='\n'+pad+'else\n'+self.stmt(n.iffalse,level+1)
   return s+'\n'+pad+'end'
  if isinstance(n,A.For):
   if isinstance(n.init,A.DeclList):assert len(n.init.decls)==1;d=n.init.decls[0];var=d.name;start=self.exp(d.init)
   else:assert isinstance(n.init,A.Assignment);var=n.init.lvalue.name;start=self.exp(n.init.rvalue)
   assert isinstance(n.cond,A.BinaryOp)and isinstance(n.cond.left,A.ID)and n.cond.left.name==var and n.cond.op in ['<','<=']
   end=self.exp(n.cond.right)
   if n.cond.op=='<':end=f'({end}-Int32(1))'
   if isinstance(n.next,A.Assignment):assert n.next.op=='+=';step=self.exp(n.next.rvalue)
   else:assert n.next.op in ['++','p++'];step='Int32(1)'
   return pad+f'for {var} in {start}:{step}:{end}\n'+self.stmt(n.stmt,level+1)+'\n'+pad+'end'
  if isinstance(n,A.Return):return pad+'return '+(self.ret+'('+self.exp(n.expr)+')'if n.expr else'nothing')
  if isinstance(n,A.Continue):return pad+'continue'
  if isinstance(n,A.Break):return pad+'break'
  if isinstance(n,A.EmptyStatement):return''
  return pad+self.exp(n)
output=['# Generated from preserved original GPU source by reviewed typed C AST translation.','# Per-thread roots/weights/G preserve row-major private storage; no oracle data embedded.','using StaticArrays','@inline xl_truth(x)=x!=0','@inline xl_pow(x,y)=x^y','struct XLPtr{A}; array::A; offset::Int32; end','@inline Base.getindex(p::XLPtr,i::Integer)=@inbounds p.array[p.offset+i]','@inline Base.setindex!(p::XLPtr,v,i::Integer)=(@inbounds p.array[p.offset+i]=v)']
for f in functions:
 name=f.decl.name;g=Generator(f);params=[x.name for x in f.decl.type.args.params];is_kernel=name.startswith('cuda_mat_');output.append(('function 'if is_kernel else'@inline function ')+name+'('+','.join(params)+')');output.append('    @inbounds begin');output.append(g.stmt(f.body,2));output.append('    end');output.append('    return nothing'if g.ret=='Nothing'else'');output.append('end\n')
z.output.write_text('\n'.join(output)+'\n');print(json.dumps({'functions':len(functions),'GPUentries':[f.decl.name for f in functions if f.decl.name.startswith('cuda_mat_')],'source_hashes':{x:hashlib.sha256((P/'reference_history'/x).read_bytes()).hexdigest()for x in ['cuda_rys_sp.cu','cuda_rys_dp.cu']},'output_sha256':hashlib.sha256(z.output.read_bytes()).hexdigest()}))
