"""Statement-preserving source AST translation of seven owned qualified MMCSF kernels.
Bounded source indices are represented as Int64; actualdomain sourceuint32 fits.
All Float32 compound multiply-adds use explicit muladd matching nvcc default.
"""
from pathlib import Path
import re,json,hashlib
from pycparser import c_parser,c_ast as A
S=Path(__file__).resolve().parent;src=S/'configured/kernels.cu';raw=src.read_text();raw=raw[raw.index('__global__'):raw.index('void mm_native_launch')];raw=re.sub(r'//[^\n]*|/\*.*?\*/','',raw,flags=re.S);raw=raw.replace('__global__','').replace('__restrict__','');ast=c_parser.CParser().parse('typedef float DTYPE; typedef unsigned long ITYPE; typedef struct {int x;} dim3;dim3 threadIdx,blockIdx,blockDim;'+raw)
class G:
 def __init__(self):self.types={}
 def typ(self,n):
  while hasattr(n,'type')and not isinstance(n,A.IdentifierType):n=n.type
  if isinstance(n,A.IdentifierType):return 'Float32'if n.names==['DTYPE']else'Float64'if n.names==['double']else'Bool'if n.names==['bool']else'Int64'
  return 'Int64'
 def exp(self,n):
  if isinstance(n,A.ID):return {'true':'true','false':'false'}.get(n.name,n.name)
  if isinstance(n,A.Constant):return 'Int64('+n.value+')'
  if isinstance(n,A.StructRef):return {'threadIdx':'tx','blockIdx':'bx','blockDim':'bd'}[n.name.name]
  if isinstance(n,A.ArrayRef):return self.exp(n.name)+'[Int('+self.exp(n.subscript)+')+1]'
  if isinstance(n,A.BinaryOp):return '('+self.exp(n.left)+' '+n.op+' '+self.exp(n.right)+')'
  if isinstance(n,A.UnaryOp):
   if n.op in ['p++','++']:return self.exp(n.expr)+' += 1'
   if n.op in ['p--','--']:return self.exp(n.expr)+' -= 1'
   return '('+n.op+self.exp(n.expr)+')'
  if isinstance(n,A.Assignment):
   left=self.exp(n.lvalue);typ=self.types.get(left)
   if n.op=='+=' and typ=='Float32' and isinstance(n.rvalue,A.BinaryOp)and n.rvalue.op=='*':return left+' = muladd('+self.exp(n.rvalue.left)+','+self.exp(n.rvalue.right)+','+left+')'
   right=self.exp(n.rvalue)
   if typ and n.op=='=':right=typ+'('+right+')'
   return left+' '+n.op+' '+right
  if isinstance(n,A.FuncCall):
   assert n.name.name=='atomicAdd';dest,val=n.args.exprs;assert isinstance(dest,A.UnaryOp)and dest.op=='&';ref=dest.expr;return 'mm_atomic!('+self.exp(ref.name)+','+self.exp(ref.subscript)+',Float64('+self.exp(val)+'))'
  raise TypeError(type(n))
 def stmt(self,n,lev=1):
  pad='    '*lev
  if n is None or isinstance(n,A.EmptyStatement):return ''
  if isinstance(n,A.Compound):return '\n'.join(self.stmt(x,lev)for x in n.block_items or [])
  if isinstance(n,A.Decl):
   typ=self.typ(n.type);self.types[n.name]=typ;return pad+n.name+' = '+typ+'('+ (self.exp(n.init)if n.init else 'false'if typ=='Bool'else'0')+')'
  if isinstance(n,A.DeclList):return '\n'.join(self.stmt(x,lev)for x in n.decls)
  if isinstance(n,A.If):return pad+'if '+self.exp(n.cond)+'\n'+self.stmt(n.iftrue,lev+1)+ ('\n'+pad+'else\n'+self.stmt(n.iffalse,lev+1)if n.iffalse else'')+'\n'+pad+'end'
  if isinstance(n,A.For):return self.stmt(n.init,lev)+'\n'+pad+'while '+self.exp(n.cond)+'\n'+self.stmt(n.stmt,lev+1)+'\n'+self.stmt(n.next,lev+1)+'\n'+pad+'end'
  return pad+self.exp(n)
functions=[x for x in ast.ext if isinstance(x,A.FuncDef)];assert len(functions)==7
out=['# Generated from configured/kernels.cu; source AST/source SHA in device_generation.json.', '@inline mm_atomic!(v,i,x) = (CUDA.atomic_add!(pointer(v,Int(i)+1),x);nothing)']
for i,f in enumerate(functions):
 g=G();params=[x.name for x in f.decl.type.args.params];body=g.stmt(f.body)
 out+=['@inline function mm_body'+str(i)+'('+','.join(params+['tx','bx','bd'])+')','    @inbounds begin',body,'    end','    nothing','end']
path=S.parent/'mmcsf-julia-cuda.jl/device_bodies.jl';path.write_text('\n'.join(out)+'\n');(S/'device_generation.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kernels':[f.decl.name for f in functions],'scope':'Allsevenactualsourcebodies, Float32locals/explicitmuladd; boundedsourceindicesInt64, noUInt32overflowactualdomain; FP64atomiconly. No native GPU fallback.'},indent=2)+'\n')
