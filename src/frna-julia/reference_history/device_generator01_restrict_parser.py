"""Translate preserved C recurrence statements with explicit reviewed GPU mapping.
Requires pycparser==2.22; generated Julia remains frozen with source/recipe hashes.
This generator does not generate scientific reference outputs or acceptance data.
"""
from pathlib import Path
import argparse,subprocess,re,json,hashlib
from pycparser import c_parser,c_ast as A
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=Path(__file__).resolve().parent;raw=(B/'native_kernels.cuh').read_text();raw=re.sub(r'^\s*#include.*$','',raw,flags=re.M)
preamble='''typedef int int_t; typedef int fbase_t; typedef unsigned long size_t;
typedef struct fparam *fparam_t; typedef int tab4_t[4][4][4][4];
enum {A,C,G,U};
#define DEV
#define HOST
#define GLOBAL
#define LOOP_MIN 3
#define LOOP_MAX 30
#define RT (1.987213e-3*310.15)
#define conversion_factor 100.0f
#define INF 14000
#define LOG log
'''
text=subprocess.run(['gcc','-E','-P','-x','c','-'],input=preamble+raw,text=True,capture_output=True,check=True).stdout;text=re.sub(r'\b__restrict__?\b','',text);ast=c_parser.CParser().parse(text)
class Generator:
 def __init__(self,name):self.name=name;self.depth=0
 def exp(self,n):
  if isinstance(n,A.ID):return {'A':'RNA_A','C':'RNA_C','G':'RNA_G','U':'RNA_U'}.get(n.name,n.name)
  if isinstance(n,A.Constant):
   if n.type in ['int','unsigned int','long int']:return 'Int32('+n.value.rstrip('uUlL')+')'
   if n.type in ['float','double']:return ('Float32'if n.value[-1:]in'fF'else'Float64')+'('+n.value.rstrip('fF')+')'
   raise TypeError((type(n),n.type,n.value))
  if isinstance(n,A.BinaryOp):
   x=self.exp(n.left);y=self.exp(n.right)
   if n.op in ['&&','||']:return f'(rna_truth({x}) {n.op} rna_truth({y}))'
   if n.op=='/':return f'div({x},{y})'
   if n.op=='%':return f'rem({x},{y})'
   return f'({x} {n.op} {y})'
  if isinstance(n,A.UnaryOp):
   x=self.exp(n.expr)
   if n.op=='*':return f'rna_deref({x})'
   if n.op=='&':
    if isinstance(n.expr,A.ArrayRef):return f'rna_offset({self.exp(n.expr.name)},{self.exp(n.expr.subscript)})'
    raise TypeError(('address',n))
   if n.op=='!':return f'(!rna_truth({x}))'
   if n.op in ['++','p++']:return f'({x} += Int32(1))'
   if n.op in ['--','p--']:return f'({x} -= Int32(1))'
   return f'({n.op}{x})'
  if isinstance(n,A.ArrayRef):return f'{self.exp(n.name)}[{self.exp(n.subscript)}]'
  if isinstance(n,A.StructRef):return f'{self.exp(n.name)}.{n.field.name}'
  if isinstance(n,A.TernaryOp):return f'(rna_truth({self.exp(n.cond)}) ? {self.exp(n.iftrue)} : {self.exp(n.iffalse)})'
  if isinstance(n,A.Assignment):
   rhs=self.exp(n.rvalue)
   if isinstance(n.lvalue,A.ID)and n.lvalue.name=='ktmpstart'and self.name=='calc_coaxial':rhs=f'({rhs}+lane)'
   return f'({self.exp(n.lvalue)} {n.op} {rhs})'
  if isinstance(n,A.FuncCall):
   name=n.name.name;args=n.args.exprs if n.args else []
   if name=='free_energy_min':
    assert isinstance(args[0],A.UnaryOp)and args[0].op=='&';lhs=self.exp(args[0].expr);return f'({lhs} = min({lhs},{self.exp(args[1])}))'
   if name in ['int_min','int_t_min','real_min']:name='min'
   return name+'('+','.join(self.exp(x)for x in args)+')'
  if isinstance(n,A.Cast):return 'Int32('+self.exp(n.expr)+')'
  if isinstance(n,A.ExprList):return '('+';'.join(self.exp(x)for x in n.exprs)+')'
  raise TypeError(('expression',type(n),n))
 def statement(self,n,level=1):
  pad='    '*level
  if n is None:return ''
  if isinstance(n,A.Compound):return '\n'.join(self.statement(x,level)for x in n.block_items or [])
  if isinstance(n,A.Decl):
   if n.name=='i'and self.name!='calc_w5_and_w3':return ''
   initial=self.exp(n.init)if n.init else 'rna_invalid(v)'if isinstance(n.type,A.PtrDecl)else'Int32(0)'
   overrides={}
   if self.name=='calc_V_bulge_internal':overrides={'d1start':'tx','d1inc':'Int32(16)','d2start':'(d1>0 ? ty : ty+Int32(1))','d2inc':'Int32(16)'}
   if self.name=='calc_WM':overrides={'kstart':'i+lane','kinc':'Int32(256)'}
   if self.name=='calc_wl_coax':overrides={'kstart':'i+Int32(4)+lane','kinc':'Int32(256)'}
   if self.name=='calc_coaxial':overrides={'ktmpinc':'Int32(256)'}
   if self.name=='calc_w5_and_w3':overrides={'istart':'lane','iinc':'Int32(256)'}
   return pad+n.name+' = '+overrides.get(n.name,initial)
  if isinstance(n,A.If):
   text=pad+'if rna_truth('+self.exp(n.cond)+')\n'+self.statement(n.iftrue,level+1)
   if n.iffalse:text+='\n'+pad+'else\n'+self.statement(n.iffalse,level+1)
   return text+'\n'+pad+'end'
  if isinstance(n,A.For):
   assert isinstance(n.init,A.Assignment)and isinstance(n.init.lvalue,A.ID);var=n.init.lvalue.name;start=self.exp(n.init.rvalue);cond=n.cond;assert isinstance(cond,A.BinaryOp)and cond.op in ['<','<='];stop=self.exp(cond.right)
   if isinstance(cond.left,A.BinaryOp):assert self.name=='calc_w5_and_w3';stop=f'({stop}-Int32(3))'
   if cond.op=='<':stop=f'({stop}-Int32(1))'
   if isinstance(n.next,A.Assignment):assert n.next.op=='+=';step=self.exp(n.next.rvalue)
   else:assert isinstance(n.next,A.UnaryOp)and n.next.op in ['p++','++'];step='Int32(1)'
   self.depth+=1;body=self.statement(n.stmt,level+1);self.depth-=1
   return pad+f'for {var} in {start}:{step}:{stop}\n'+body+'\n'+pad+'end'
  if isinstance(n,A.Continue):
   if self.depth:return pad+'continue'
   defaults={'calc_V_bulge_internal':'RNA_INF','calc_WM':'(RNA_INF,false)','calc_coaxial':'RNA_INF','calc_wl_coax':'(RNA_INF,RNA_INF,false)'}
   return pad+'return '+defaults.get(self.name,'nothing')
  if isinstance(n,A.Break):return pad+'break'
  if isinstance(n,A.Return):return pad+'return '+(self.exp(n.expr)if n.expr else'nothing')
  if isinstance(n,A.EmptyStatement):return ''
  return pad+self.exp(n)
functions={x.decl.name:x for x in ast.ext if isinstance(x,A.FuncDef)and x.decl.name.startswith('calc_')};assert len(functions)==9
output=['# Generated recurrence statements from exact native source; reviewed per-cell/partial mappings.','# See generate_device_bodies.py and native_kernels.cuh; GPL-2.0-or-later original source retained.']
for name,f in functions.items():
 gen=Generator(name);params=[x.name for x in f.decl.type.args.params];nodes=f.body.block_items
 if name=='calc_w5_and_w3':
  body=nodes[:-4];args=params+['lane'];tail='return w5tmp,w3tmp'
 else:
  outer=next(x for x in nodes if isinstance(x,A.For));body=outer.stmt.block_items;args=['i']+params
  if name=='calc_V_bulge_internal':body=body[:-1];args+=['tx','ty'];tail='return vij'
  elif name=='calc_WM':body=body[:-2];args+=['lane'];tail='return tmp,true'
  elif name=='calc_coaxial':body=body[:-1];args+=['lane'];tail='return vij'
  elif name=='calc_wl_coax':body=body[:-4];args+=['lane'];tail='return tmp1,tmp2,true'
  else:tail='return nothing'
 output+=['@inline function '+name+'_body('+','.join(args)+')']+[gen.statement(x)for x in body]+['    '+tail,'end','']
a.output.write_text('\n'.join(output)+'\n');print(json.dumps({'generator':'pycparser2.22/sourceCPUstatements with reviewedparallelstart/stride andexplicitreductionreturn mappings','functions':list(functions),'source_sha256':hashlib.sha256((B/'native_kernels.cuh').read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))
