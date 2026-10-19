"""Source-bound C AST -> Julia translation; emits only requested target phase.
Parser: existing isolated pycparser 2.22, PYTHONPATH=/tmp/frna_parser_deps.
No algorithm substitution: zero-based pointer views, original control/barriers/types.
"""
import json,re,sys,hashlib
from pathlib import Path
from pycparser import c_parser,c_ast
import pycparser
P=Path(__file__).resolve().parent; ROOT=P.parent.parent
model=sys.argv[1]; assert model in ['cuda_jl','ka','jacc']
source=ROOT/'src/heartwall-cuda/kernel/kernel.h'; raw=source.read_text()
s=re.sub(r'/\*.*?\*/|//[^\n]*','',raw,flags=re.S)
s=re.sub(r'#ifdef TEST_CHECKSUM\b.*?#endif','',s,flags=re.S)
s=s.replace('__global__','').replace('__restrict__','')
fields=json.loads((P/'params_schema.json').read_text())['fields']
prefix='typedef float fp; typedef struct {'+';'.join(('float' if f['type']=='float32' else 'int')+' '+f['name'] for f in fields)+';} params_common;\n'
ast=c_parser.CParser().parse(prefix+s)
fun=ast.ext[-1]
class Emit:
 def __init__(self): self.types={f'd_common.{f["name"]}':('f' if f['type']=='float32' else 'i') for f in fields};self.types.update(frame_no='i',NUMBER_THREADS='i')
 def typ(self,n):
  if isinstance(n,c_ast.ID):return self.types.get(n.name,'i')
  if isinstance(n,c_ast.Constant):return 'f' if n.type in ('float','double') else 'i'
  if isinstance(n,c_ast.StructRef):return self.types.get(self.expr(n),'i')
  if isinstance(n,c_ast.ArrayRef):return self.typ(n.name).replace('p','')
  if isinstance(n,c_ast.UnaryOp):return self.typ(n.expr)
  if isinstance(n,c_ast.Cast):return 'f' if self.typename(n.to_type.type)=='Float32' else 'i'
  if isinstance(n,c_ast.BinaryOp):return 'f' if 'f' in [self.typ(n.left),self.typ(n.right)] else 'i'
  if isinstance(n,c_ast.TernaryOp):return self.typ(n.iftrue)
  if isinstance(n,c_ast.FuncCall):return 'f'
  raise TypeError(type(n))
 def typename(self,n):
  while hasattr(n,'type') and not isinstance(n,c_ast.IdentifierType): n=n.type
  return {'int':'Int32','fp':'Float32','float':'Float32','params_common':'Any'}[' '.join(n.names)]
 def expr(self,n):
  if isinstance(n,c_ast.ID):return 'Int32(256)' if n.name=='NUMBER_THREADS' else n.name
  if isinstance(n,c_ast.Constant):return ('Float32('+n.value.rstrip('fF')+')') if n.type in ('float','double') else 'Int32('+n.value+')'
  if isinstance(n,c_ast.StructRef):
   if isinstance(n.name,c_ast.ID) and n.name.name in ('blockIdx','threadIdx'):return {'blockIdx':'bx0','threadIdx':'tx0'}[n.name.name]
   return self.expr(n.name)+'.'+n.field.name
  if isinstance(n,c_ast.ArrayRef):return self.expr(n.name)+'['+self.expr(n.subscript)+']'
  if isinstance(n,c_ast.UnaryOp):
   if n.op=='&':assert isinstance(n.expr,c_ast.ArrayRef);return 'shiftview('+self.expr(n.expr.name)+','+self.expr(n.expr.subscript)+')'
   if n.op in ('p++','++'):return self.expr(n.expr)+' += Int32(1)'
   if n.op in ('p--','--'):return self.expr(n.expr)+' -= Int32(1)'
   return '('+n.op+self.expr(n.expr)+')'
  if isinstance(n,c_ast.BinaryOp):
   a,b=self.expr(n.left),self.expr(n.right)
   if n.op=='/' and self.typ(n)=='i':return 'div('+a+','+b+')'
   # NVCC default contraction of a product followed by sum/subtraction.
   if n.op in ('+','-') and self.typ(n)=='f':
    if isinstance(n.right,c_ast.BinaryOp) and n.right.op=='*':
     v,w=self.expr(n.right.left),self.expr(n.right.right)
     return 'fma(Float32('+('-' if n.op=='-' else '')+v+'),Float32('+w+'),Float32('+a+'))'
    if n.op=='+' and isinstance(n.left,c_ast.BinaryOp) and n.left.op=='*':return 'fma(Float32('+self.expr(n.left.left)+'),Float32('+self.expr(n.left.right)+'),Float32('+b+'))'
   return '('+a+' '+n.op+' '+b+')'
  if isinstance(n,c_ast.TernaryOp):return '('+self.expr(n.cond)+' ? '+self.expr(n.iftrue)+' : '+self.expr(n.iffalse)+')'
  if isinstance(n,c_ast.Cast):return self.typename(n.to_type.type)+'('+self.expr(n.expr)+')'
  if isinstance(n,c_ast.FuncCall):return {'fabs':'abs','sqrt':'sqrt'}.get(n.name.name,n.name.name)+'('+','.join(self.expr(x) for x in (n.args.exprs if n.args else []))+')'
  if isinstance(n,c_ast.Assignment):return self.expr(n.lvalue)+' '+n.op+' '+self.expr(n.rvalue)
  raise TypeError(type(n))
 def stmt(self,n,lev=1):
  pad='    '*lev
  if n is None:return ''
  if isinstance(n,c_ast.Compound):return ''.join(self.stmt(x,lev) for x in n.block_items or [])
  if isinstance(n,c_ast.Decl):
   ptr=isinstance(n.type,c_ast.PtrDecl);typ=self.typename(n.type)
   self.types[n.name]=('f' if typ=='Float32' else 'i')+('p' if ptr else '')
   return pad+('' if ptr else 'local ')+n.name+('' if ptr else '::'+typ)+((' = '+self.expr(n.init)) if n.init else '')+'\n'
  if isinstance(n,c_ast.If):return pad+'if '+self.expr(n.cond)+'\n'+self.stmt(n.iftrue,lev+1)+(pad+'else\n'+self.stmt(n.iffalse,lev+1) if n.iffalse else '')+pad+'end\n'
  if isinstance(n,(c_ast.While,c_ast.For)):
   init=self.stmt(n.init,lev) if isinstance(n,c_ast.For) else ''
   nex=self.stmt(n.next,lev+1) if isinstance(n,c_ast.For) else ''
   return init+pad+'while '+self.expr(n.cond)+'\n'+self.stmt(n.stmt,lev+1)+nex+pad+'end\n'
  if isinstance(n,c_ast.FuncCall) and n.name.name=='__syncthreads':return pad+{'cuda_jl':'CUDA.sync_threads()','ka':'@synchronize()','jacc':'JACC.sync_workgroup()'}[model]+'\n'
  if isinstance(n,c_ast.EmptyStatement):return ''
  if isinstance(n,c_ast.DeclList):return ''.join(self.stmt(x,lev) for x in n.decls)
  return pad+self.expr(n)+'\n'
e=Emit()
params=fun.decl.type.args.params
for p in params:
 if isinstance(p.type,c_ast.PtrDecl): e.types[p.name]=('f' if e.typename(p.type)=='Float32' else 'i')+'p'
body=e.stmt(fun.body)
head='''# Generated from the unchanged HEARTWALL C kernel AST; see translation_provenance.json.
struct ZeroView{A}
    data::A
    offset::Int32
end
@inline Base.getindex(v::ZeroView, i::Integer) = @inbounds v.data[v.offset+i+Int32(1)]
@inline Base.setindex!(v::ZeroView, x, i::Integer) = (@inbounds v.data[v.offset+i+Int32(1)]=x)
@inline shiftview(v::ZeroView,i::Integer)=ZeroView(v.data,v.offset+Int32(i))
'''
args=[p.name for p in params if isinstance(p.type,c_ast.PtrDecl)]
setup=''.join('    '+a+' = ZeroView(buffers.'+a[2:]+',Int32(16))\n' for a in args)
start={'cuda_jl':'function hw(frame_no::Int32,d_common,buffers)\n    bx0=Int32(CUDA.blockIdx().x-1);tx0=Int32(CUDA.threadIdx().x-1)\n','ka':'@kernel function hw(frame_no::Int32,d_common,buffers)\n    bx0=Int32(@index(Group,Linear))-Int32(1);tx0=Int32(@index(Local,Linear))-Int32(1)\n','jacc':'function hw(global_i,frame_no::Int32,d_common,buffers)\n    bx0=Int32(div(global_i-1,256));tx0=Int32(rem(global_i-1,256))\n'}[model]
child=P/('heartwall-julia-'+{'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}[model]);child.mkdir(exist_ok=True)
(child/'kernel.jl').write_text(head+start+setup+body+'    nothing\nend\n')
(child/'translation_provenance.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parser':pycparser.__version__,'parser_path':str(Path(pycparser.__file__).resolve()),'policy':'sourceAST all non-checksum statements, typed Int32/Float32 zero-based guarded views; integer div truncates toward zero; explicit float product-add contraction, unchanged barriers','model':model,'kernel_definitions':1},indent=2)+'\n')
print(child/'kernel.jl')
