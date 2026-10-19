# Typed emitter core derived from frozen WSM5 generator; STSGextensions in translate_filter.py.
alpha_map=[];serial=[0]
def bindwalk(n,scopes):
 if n is None:return
 if isinstance(n,c_ast.Compound):
  inner=scopes+[{}]
  for child in n.block_items or[]:bindwalk(child,inner)
  return
 if isinstance(n,c_ast.For):
  inner=scopes+[{}]
  for child in [n.init,n.cond,n.next,n.stmt]:bindwalk(child,inner)
  return
 if isinstance(n,c_ast.Decl):
  original=n.name;serial[0]+=1;new=original+'_c'+str(serial[0]);scopes[-1][original]=new;n.name=new;alpha_map.append({'source_name':original,'target_name':new,'preprocessed_coord':str(n.coord)})
  bindwalk(n.init,scopes)
  return
 if isinstance(n,c_ast.ID):
  for scope in reversed(scopes):
   if n.name in scope:n.name=scope[n.name];break
  return
 if isinstance(n,c_ast.StructRef):bindwalk(n.name,scopes);return
 for _,child in n.children():bindwalk(child,scopes)
paramscope={x.name:x.name for x in fn.decl.type.args.params}
bindwalk(fn.body,[paramscope])
types={};arrays=set();calls={'sqrtf':'Float32','logf':'Float32','expf':'Float32','powf':'Float32','rsqrtf':'Float32','abs':'Float32','sqrt':'Float64','pow':'Float64','trunc':'Float64'}
def ctype(t):
 if isinstance(t,c_ast.TypeDecl):return ctype(t.type)
 if isinstance(t,c_ast.IdentifierType):
  s=' '.join(t.names);return {'float':'Float32','double':'Float64','int':'Int32','unsigned int':'UInt32','uint32_t':'UInt32','short':'Int16','unsigned char':'UInt8'}[s]
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
