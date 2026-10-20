"""PRNA-only C statement extractor; source-derived expressions, reviewed GPU partitions.
Uses existing isolated pycparser2.22. No other benchmark/tool file is written.
"""
from pathlib import Path
import argparse,subprocess,re,json,hashlib
from pycparser import c_parser,c_ast as A
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);opt=parser.parse_args();B=Path(__file__).resolve().parent
assert opt.output.resolve().is_relative_to(B), 'Only PRNA-owned output is permitted'
raw=(B/'native_kernels.cuh').read_text();raw=re.sub(r'^\s*#include.*$','',raw,flags=re.M)
preamble='''typedef float real_t;typedef int base_t;typedef unsigned long size_t;typedef struct param *param_t;typedef float tab4_t[5][5][5][5];
enum{X,A,C,G,U};
#define DEV
#define HOST
#define GLOBAL
#define RCONST(x) x
#define LOOP_MIN 3
#define LOOP_MAX 30
#define INF RNA_INF
#define LOG logf
#define LOG1P log1pf
#define EXP expf
'''
text=subprocess.run(['gcc','-E','-P','-x','c','-'],input=preamble+raw,text=True,capture_output=True,check=True).stdout
text=re.sub(r'\b__restrict(?:__)?\b','',text);ast=c_parser.CParser().parse(text)
class Generator:
 def __init__(self,name):self.name=name;self.depth=0
 def exp(self,n):
  if isinstance(n,A.ID):return {'X':'RNA_X','A':'RNA_A','C':'RNA_C','G':'RNA_G','U':'RNA_U'}.get(n.name,n.name)
  if isinstance(n,A.Constant):
   if n.type in ['int','unsigned int','long int']:return 'Int32('+n.value.rstrip('uUlL')+')'
   if n.type in ['float','double']:return ('Float32' if n.value[-1:] in 'fF' else 'Float64')+'('+n.value.rstrip('fF')+')'
   raise TypeError(('constant',n.type,n.value))
  if isinstance(n,A.BinaryOp):
   x,y=self.exp(n.left),self.exp(n.right)
   if n.op in ['&&','||']:return f'(rna_truth({x}) {n.op} rna_truth({y}))'
   if n.op=='/':return f'div({x},{y})'
   if n.op=='%':return f'rem({x},{y})'
   return f'({x} {n.op} {y})'
  if isinstance(n,A.UnaryOp):
   x=self.exp(n.expr)
   if n.op=='*':return f'rna_deref({x})'
   if n.op=='&':
    assert isinstance(n.expr,A.ArrayRef)
    return f'rna_offset({self.exp(n.expr.name)},{self.exp(n.expr.subscript)})'
   if n.op=='!':return f'(!rna_truth({x}))'
   if n.op in ['++','p++']:return f'({x} += Int32(1))'
   if n.op in ['--','p--']:return f'({x} -= Int32(1))'
   return f'({n.op}{x})'
  if isinstance(n,A.ArrayRef):return f'{self.exp(n.name)}[{self.exp(n.subscript)}]'
  if isinstance(n,A.StructRef):return f'{self.exp(n.name)}.{n.field.name}'
  if isinstance(n,A.TernaryOp):return f'(rna_truth({self.exp(n.cond)}) ? {self.exp(n.iftrue)} : {self.exp(n.iffalse)})'
  if isinstance(n,A.Assignment):
   rhs=self.exp(n.rvalue)
   if isinstance(n.lvalue,A.ID)and n.lvalue.name in ['ktmpstart','kstart']and self.name=='calc_coaxial':rhs=f'({rhs}+lane)'
   lhs=self.exp(n.lvalue);return f'(begin {lhs} {n.op} {rhs}; {lhs} end)'
  if isinstance(n,A.FuncCall):
   name=n.name.name;args=n.args.exprs if n.args else []
   if name=='free_energy_accumulate':
    assert isinstance(args[0],A.UnaryOp)and args[0].op=='&'
    lhs=self.exp(args[0].expr);return f'({lhs} = free_energy_sum({lhs},{self.exp(args[1])}))'
   if name in ['int_min','real_min']:name='min'
   return name+'('+','.join(self.exp(x)for x in args)+')'
  if isinstance(n,A.Cast):return 'Float32('+self.exp(n.expr)+')'
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
   if self.name=='calc_internal':overrides={'d1start':'tx','d1inc':'Int32(8)','d2start':'(d1>0 ? ty : ty+Int32(1))','d2inc':'Int32(16)'}
   if self.name in ['calc_xl','calc_z']:overrides={'kstart':'('+initial+'+lane)','kinc':'Int32(128)'}
   if self.name=='calc_coaxial':overrides={'ktmpinc':'Int32(128)','ktmpstart':'('+initial+'+lane)','kinc':'Int32(128)'}
   if self.name=='calc_w5_and_w3':overrides={'istart':'lane','iinc':'Int32(128)'}
   return pad+n.name+' = '+overrides.get(n.name,initial)
  if isinstance(n,A.If):
   result=pad+'if rna_truth('+self.exp(n.cond)+')\n'+self.statement(n.iftrue,level+1)
   if n.iffalse:result+='\n'+pad+'else\n'+self.statement(n.iffalse,level+1)
   return result+'\n'+pad+'end'
  if isinstance(n,A.For):
   assert isinstance(n.init,A.Assignment)and isinstance(n.init.lvalue,A.ID)
   var=n.init.lvalue.name;start=self.exp(n.init.rvalue);cond=n.cond;assert isinstance(cond,A.BinaryOp)and cond.op in ['<','<='];stop=self.exp(cond.right)
   if isinstance(cond.left,A.BinaryOp):assert self.name=='calc_w5_and_w3';stop=f'({stop}-Int32(3))'
   if cond.op=='<':stop=f'({stop}-Int32(1))'
   if isinstance(n.next,A.Assignment):assert n.next.op=='+=';step=self.exp(n.next.rvalue)
   else:assert isinstance(n.next,A.UnaryOp)and n.next.op in ['p++','++'];step='Int32(1)'
   self.depth+=1;body=self.statement(n.stmt,level+1);self.depth-=1
   return pad+f'for {var} in {start}:{step}:{stop}\n'+body+'\n'+pad+'end'
  if isinstance(n,A.Continue):
   if self.depth:return pad+'continue'
   defaults={'calc_internal':'RNA_INF','calc_coaxial':'RNA_INF','calc_xl':'RNA_INF','calc_z':'(RNA_INF,RNA_INF)'}
   return pad+'return '+defaults.get(self.name,'nothing')
  if isinstance(n,A.Break):return pad+'break'
  if isinstance(n,A.Return):return pad+'return '+(self.exp(n.expr)if n.expr else'nothing')
  if isinstance(n,A.EmptyStatement):return ''
  return pad+self.exp(n)
functions={x.decl.name:x for x in ast.ext if isinstance(x,A.FuncDef)and x.decl.name.startswith('calc_')};assert len(functions)==8
output=['# Source-derived PRNA recurrence statements. Explicit GPU partitions/reduction tree are separate.']
for name,f in functions.items():
 gen=Generator(name);params=[x.name for x in f.decl.type.args.params];nodes=f.body.block_items
 if name=='calc_w5_and_w3':body=nodes[:-4];args=params+['lane'];tail='return w5tmp,w3tmp'
 else:
  outer=next(x for x in nodes if isinstance(x,A.For));body=list(outer.stmt.block_items);args=['i']+params
  if name=='calc_internal':body=body[:-1];args+=['tx','ty'];tail='return vij'
  elif name=='calc_coaxial':body=body[:-1];args+=['lane'];tail='return vij'
  elif name=='calc_xl':
   body=[x for x in body[:-1] if not(isinstance(x,A.Assignment)and isinstance(x.lvalue,A.ArrayRef)and isinstance(x.lvalue.name,A.ID)and x.lvalue.name.name=='xl')];args+=['lane'];tail='return tmp'
  elif name=='calc_z':body=body[:-4];args+=['lane'];tail='return tmp1,tmp2'
  else:tail='return nothing'
 output+=['@inline function '+name+'_body('+','.join(args)+')']+[gen.statement(x)for x in body]+['    '+tail,'end','']
opt.output.write_text('\n'.join(output)+'\n')
print(json.dumps({'generator':'sourceCPUstatements plusreviewedoriginalGPUstart/stride/reductionreturn mappings','functions':list(functions),'source_sha256':hashlib.sha256((B/'native_kernels.cuh').read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(opt.output.read_bytes()).hexdigest()}))
