"""Source C AST -> explicit typed Julia operator. No reduced physics or vendor fallback."""
from pathlib import Path
import sys,subprocess,json,hashlib
sys.path.insert(0,'/tmp/wsm5-python-tools')
from pycparser import c_parser,c_ast
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
pre=subprocess.check_output(['gcc','-E','-P','-x','c','-I',str(R/'src/wsm5-cuda'),str(p/'study_parse_input.c')],text=True);(p/'study_preprocessed_operator.c').write_text(pre);ast=c_parser.CParser().parse(pre);fn=next(x for x in ast.ext if isinstance(x,c_ast.FuncDef)and x.decl.name=='wsm')
types={};arrays=set();calls={'sqrtf':'Float32','logf':'Float32','expf':'Float32','powf':'Float32','rsqrtf':'Float32','abs':'Float32','sqrt':'Float64','pow':'Float64','trunc':'Float64'}
def ctype(t):
 if isinstance(t,c_ast.TypeDecl):return ctype(t.type)
 if isinstance(t,c_ast.IdentifierType):
  s=' '.join(t.names);return {'float':'Float32','double':'Float64','int':'Int32','unsigned int':'UInt32','uint32_t':'UInt32'}[s]
 if isinstance(t,c_ast.PtrDecl):return ('ptr',ctype(t.type))
 if isinstance(t,c_ast.ArrayDecl):return ('array',ctype(t.type),int(t.dim.value))
 raise RuntimeError(('ctype',t))
def scan(n):
 if isinstance(n,c_ast.Decl)and not isinstance(n.type,c_ast.FuncDecl):
  t=ctype(n.type)
  if n.name in types:assert types[n.name]==t,(n.name,types[n.name],t)
  types[n.name]=t
 for _,v in n.children():scan(v)
scan(fn)
def promote(a,b):
 assert isinstance(a,str)and isinstance(b,str),(a,b)
 if 'Float64'in(a,b):return 'Float64'
 if 'Float32'in(a,b):return 'Float32'
 if 'UInt32'in(a,b):return 'UInt32'
 return 'Int32'
def typex(n):
 if isinstance(n,c_ast.ID):return types[n.name]
 if isinstance(n,c_ast.StructRef):return 'UInt32'
 if isinstance(n,c_ast.ArrayRef):return typex(n.name)[1]
 if isinstance(n,c_ast.Constant):return {'float':'Float32','double':'Float64','int':'Int32','unsigned int':'UInt32','char':'Int32'}[n.type]
 if isinstance(n,c_ast.Cast):return ctype(n.to_type.type)
 if isinstance(n,c_ast.FuncCall):return calls[n.name.name]
 if isinstance(n,c_ast.UnaryOp):return 'Bool'if n.op=='!'else typex(n.expr)
 if isinstance(n,c_ast.BinaryOp):
  if n.op in ['<','>','<=','>=','==','!=','&&','||']:return 'Bool'
  return promote(typex(n.left),typex(n.right))
 if isinstance(n,c_ast.TernaryOp):return promote(typex(n.iftrue),typex(n.iffalse))
 if isinstance(n,c_ast.Assignment):return typex(n.lvalue)
 raise RuntimeError(('typex',type(n).__name__,n))
def conv(t,s,source=None):
 if t in ['Int32','UInt32']and source in ['Float32','Float64']:return f'unsafe_trunc({t},{s})'
 return f'{t}({s})'
def condition(n):
 s=ex(n);return s if typex(n)=='Bool'else f'(!iszero({s}))'
def ex(n):
 if isinstance(n,c_ast.ID):return n.name
 if isinstance(n,c_ast.StructRef):
  return {('blockIdx','x'):'bidx',('blockIdx','y'):'bidy',('blockDim','x'):'bdimx',('blockDim','y'):'bdimy',('threadIdx','x'):'tidx',('threadIdx','y'):'tidy'}[(n.name.name,n.field.name)]
 if isinstance(n,c_ast.ArrayRef):return f'{ex(n.name)}[Int({ex(n.subscript)})+1]'
 if isinstance(n,c_ast.Constant):
  t=typex(n);v=n.value
  if t=='Float32':v=v[:-1]if v.lower().endswith('f')else v
  if t=='UInt32':v=v.rstrip('uUlL')
  if v.startswith('.'):v='0'+v
  return f'{t}({v})'
 if isinstance(n,c_ast.Cast):
  t=ctype(n.to_type.type)
  if t in ['Int32','UInt32']and isinstance(n.expr,c_ast.FuncCall)and n.expr.name.name=='trunc':return conv(t,ex(n.expr.args.exprs[0]),typex(n.expr.args.exprs[0]))
  return conv(t,ex(n.expr),typex(n.expr))
 if isinstance(n,c_ast.FuncCall):
  name=n.name.name;args=n.args.exprs if n.args else[]
  if name in ['sqrtf','logf','expf','powf','rsqrtf','abs']:
   v=[conv('Float32',ex(x),typex(x))for x in args]
   if name=='powf':s=f'({v[0]}^{v[1]})'
   elif name=='rsqrtf':s=f'CUDA.rsqrt({v[0]})'
   else:s={'sqrtf':'sqrt','logf':'log','expf':'exp','abs':'abs'}[name]+'('+','.join(v)+')'
   return conv('Float32',s)
  v=[conv('Float64',ex(x),typex(x))for x in args]
  if name=='pow':s=f'({v[0]}^{v[1]})'
  else:s=name+'('+','.join(v)+')'
  return conv('Float64',s)
 if isinstance(n,c_ast.UnaryOp):
  if n.op=='!':return '(!'+condition(n.expr)+')'
  if n.op in ['+','-','~']:return conv(typex(n),f'({n.op}{ex(n.expr)})')
  if n.op in ['p++','++','p--','--']:return ex(n.expr)+(' += 'if'+'in n.op else' -= ')+conv(typex(n.expr),'1')
  raise RuntimeError(('unary',n.op))
 if isinstance(n,c_ast.BinaryOp):
  a=ex(n.left);b=ex(n.right);op=n.op
  if op in ['&&','||']:return '('+condition(n.left)+op+condition(n.right)+')'
  if op in ['<','>','<=','>=','==','!=']:return f'({a}{op}{b})'
  t=typex(n)
  if op=='/'and t in ['Int32','UInt32']:s=f'div({a},{b})'
  elif op=='%':s=f'rem({a},{b})'
  else:s=f'({a}{op}{b})'
  return conv(t,s)
 if isinstance(n,c_ast.TernaryOp):return conv(typex(n),'('+condition(n.cond)+' ? '+ex(n.iftrue)+' : '+ex(n.iffalse)+')')
 if isinstance(n,c_ast.Assignment):
  t=typex(n.lvalue);right=ex(n.rvalue)
  if n.op=='=':return ex(n.lvalue)+' = '+conv(t,right,typex(n.rvalue))
  return ex(n.lvalue)+' '+n.op+' '+right
 raise RuntimeError(('ex',type(n).__name__,n))
lines=[];decls=[]
def emit(s,level):lines.append(' '*level+s)
def stmt(n,level):
 if n is None:return
 if isinstance(n,c_ast.Compound):
  for x in n.block_items or[]:stmt(x,level)
 elif isinstance(n,c_ast.Decl):
  t=types[n.name];decls.append((n.name,t))
  if isinstance(t,tuple):
   assert t[0]=='array';v=','.join([t[1]+'(NaN)']*t[2]);emit(f'local {n.name} = MVector{{{t[2]},{t[1]}}}(({v}))',level)
  else:
   value=conv(t,ex(n.init),typex(n.init))if n.init is not None else (t+'(NaN)'if t.startswith('Float')else t+'(0)')
   emit(f'local {n.name}::{t} = {value}',level)
 elif isinstance(n,c_ast.If):
  emit('if '+condition(n.cond),level);stmt(n.iftrue,level+1)
  if n.iffalse is not None:emit('else',level);stmt(n.iffalse,level+1)
  emit('end',level)
 elif isinstance(n,c_ast.For):
  stmt(n.init,level);emit('while '+condition(n.cond),level);stmt(n.stmt,level+1);stmt(n.next,level+1);emit('end',level)
 elif isinstance(n,c_ast.DeclList):
  for d in n.decls:stmt(d,level)
 elif isinstance(n,(c_ast.Assignment,c_ast.UnaryOp,c_ast.FuncCall)):emit(ex(n),level)
 elif isinstance(n,c_ast.EmptyStatement):pass
 else:raise RuntimeError(('stmt',type(n).__name__,n))
params=[x.name for x in fn.decl.type.args.params];emit('function wsm_body!('+','.join(params+['bidx::UInt32','bidy::UInt32','bdimx::UInt32','bdimy::UInt32','tidx::UInt32','tidy::UInt32'])+')',0);emit('@inbounds begin',1);stmt(fn.body,2);emit('end',1);emit('nothing',1);emit('end',0)
header='''# Full source C AST translation, generated by study_translate_c_ast.py.
# Float32/Float64 usual arithmetic/assignment types explicit; no reduced operator.
# Target scalar/local-array diagnostic NaNs initialize otherwise uninitialized slots;
# source-defined named paths overwrite them before use. Extra initialization mapping disclosed.
'''
(p/'study_device.jl').write_text(header+'\n'.join(lines)+'\n');(p/'translation_generation_review.json').write_text(json.dumps({'tool':'pycparser','version':__import__('pycparser').__version__,'source_sha256':hashlib.sha256((p/'study_native_kernel.h').read_bytes()).hexdigest(),'preprocessed_sha256':hashlib.sha256(pre.encode()).hexdigest(),'generated_sha256':hashlib.sha256((p/'study_device.jl').read_bytes()).hexdigest(),'parameters':params,'local_declarations':decls,'type_policy':'CusualFloat32/Float64types +assignmentnarrowing; floattointunsafe_trunc; sourceUInt32geometry. AlltargetlocalsdiagnosticNaN/0initializationdisclosed. rsqrtfusesCUDAownintrinsic hybriddependencyoutsideCUDAjl.'},indent=2)+'\n');print('generated',len(lines),'lines',len((p/'study_device.jl').read_text()),'chars')
