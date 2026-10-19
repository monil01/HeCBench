"""Typed C AST translation of complete source SHA/AES bodies, not a crypto stub.
Emitter core reused from previously audited STSG/WSM5 lineage. Requires isolated
pycparser3 under /tmp/wsm5-python-tools for regeneration; generated Julia needs
no Python parser. Source GPL notices are retained with archived originals.
"""
from pathlib import Path
import sys,re,json,hashlib,subprocess,copy
sys.path.insert(0,'/tmp/wsm5-python-tools')
from pycparser import c_parser,c_ast
P=Path(__file__).resolve().parent;O=P.parent/'bitcracker-cuda'
def comments(s):return re.sub(r'//[^\n]*','',re.sub(r'/\*.*?\*/','',s,flags=re.S))
a=comments((O/'attack.cu').read_text());a=a[a.index('__device__'):a.index('int   *h_found;')]
w=comments((O/'w_blocks.cu').read_text());w=w[w.index('#define ROR'):w.index('int evaluate_w_block(')]
sha=comments((O/'sha256.h').read_text());aes=comments((O/'aes.h').read_text())
head='''typedef unsigned int uint32_t;typedef unsigned long long uint64_t;typedef unsigned char uint8_t;typedef signed char int8_t;
typedef struct {unsigned int x,y,z;} Dim;Dim blockIdx,blockDim,threadIdx,gridDim;
#define __device__
#define __global__
#define __constant__
#define __restrict__
#define __forceinline__
#define UINT32_C(c) c##U
#define SALT_SIZE 16
#define HASH_BLOCK_NUM_UINT32 64
#define PADDING_SIZE 40
#define NUM_HASH_BLOCKS 0x100000
#define PSWD_NUM_UINT32 32
'''
a=a.replace('auto condition','int condition')
src=head+sha+'\n'+aes+'\n'+a+'\n'+w
(P/'translation_input.c').write_text(src)
pp=subprocess.run(['gcc','-E','-P','-x','c',str(P/'translation_input.c')],check=True,capture_output=True,text=True).stdout
(P/'translation_preprocessed.c').write_text(pp)
ast=c_parser.CParser().parse(pp);core=(P/'typed_emitter_core.py').read_text();outputs=[];reports=[]
# Constant tables retained as immutable tuples: no extra runtime GPU buffers.
for node in ast.ext:
 if isinstance(node,c_ast.Decl)and node.name in ['TS0','TS1','TS2','TS3']:
  assert isinstance(node.init,c_ast.InitList)and len(node.init.exprs)==256
  vals=[int(z.value.rstrip('uUlL'),0)for z in node.init.exprs]
  outputs.append('const '+node.name+' = ('+','.join('UInt32('+str(v)+')'for v in vals)+',)')
for fn in ast.ext:
 if not isinstance(fn,c_ast.FuncDef):continue
 name=fn.decl.name
 assert name in ['LOP3LUT_XOR','LOP3LUT_XORAND','LOP3LUT_ANDOR','encrypt','decrypt_vmk_with_mac','kernel_w_block'],name
 # Output pointers of encrypt become a pure four-value tuple return; no local heap arrays.
 outputargs=[]
 if name=='encrypt':
  outputargs=[q.name for q in fn.decl.type.args.params[-4:]]
  fn.decl.type.args.params=fn.decl.type.args.params[:-4]
  class Rewrite(c_ast.NodeVisitor):
   def generic_visit(self,n):
    for slot in n.__slots__:
     if slot in ['coord','__weakref__']:continue
     z=getattr(n,slot,None)
     if isinstance(z,c_ast.ArrayRef)and isinstance(z.name,c_ast.ID)and z.name.name in outputargs:
      assert isinstance(z.subscript,c_ast.Constant)and z.subscript.value=='0';setattr(n,slot,c_ast.ID(z.name.name))
     elif isinstance(z,list):
      for ii,q in enumerate(z):
       if isinstance(q,c_ast.Node):self.visit(q)
     elif isinstance(z,c_ast.Node):self.visit(z)
  Rewrite().visit(fn.body)
 env={'fn':fn,'c_ast':c_ast};exec(compile(core,'typed_emitter_core.py','exec'),env)
 oldctype=env['ctype'];oldtype=env['typex'];oldex=env['ex']
 def ctype(t):
  if isinstance(t,c_ast.TypeDecl):return ctype(t.type)
  if isinstance(t,c_ast.IdentifierType):
   n=' '.join(t.names)
   if n in ['uint64_t','unsigned long long']:return 'UInt64'
   if n in ['uint8_t']:return 'UInt8'
   if n in ['int8_t','signed char']:return 'Int8'
  return oldctype(t)
 # Original scan alreadyhandled mosttypes; extend core beforeexec below ifneeded.
 env['ctype']=ctype
 for table in ['TS0','TS1','TS2','TS3']:env['types'][table]=('array','UInt32',256)
 for q in outputargs:env['types'][q]='UInt32'
 env['calls'].update({q:'UInt32'for q in ['LOP3LUT_XOR','LOP3LUT_XORAND','LOP3LUT_ANDOR','__byte_perm']})
 def typex(n):
  if isinstance(n,c_ast.Constant)and n.type=='int'and n.value.lower().startswith('0x')and int(n.value,16)>2147483647:return 'UInt32'
  if isinstance(n,c_ast.UnaryOp)and n.op=='&':return ('ptr',typex(n.expr))
  if isinstance(n,c_ast.UnaryOp)and n.op=='*':return typex(n.expr)[1]
  return oldtype(n)
 def promote(a,b):
  assert isinstance(a,str)and isinstance(b,str),(a,b)
  if 'UInt64'in(a,b):return 'UInt64'
  if 'UInt32'in(a,b):return 'UInt32'
  return 'Int32'
 def conv(t,s,source=None):
  if t in ['UInt8','Int8','UInt32','Int32','UInt64']:return '(('+s+') % '+t+')'
  return t+'('+s+')'
 def ex(n):
  if isinstance(n,c_ast.Assignment)and n.op=='^=':return ex(n.lvalue)+' ⊻= '+ex(n.rvalue)
  if isinstance(n,c_ast.UnaryOp)and n.op=='*':return ex(n.expr)+'[1]'
  if isinstance(n,c_ast.StructRef)and(n.name.name,n.field.name)==('gridDim','x'):return 'gdimx'
  if isinstance(n,c_ast.Constant):
   v=n.value.rstrip('uUlL');return conv(typex(n),v)
  if isinstance(n,c_ast.FuncCall)and n.name.name in env['calls']:
   q=n.name.name;args=[ex(z)for z in n.args.exprs]
   if q=='__byte_perm':
    assert len(args)==3 and isinstance(n.args.exprs[1],c_ast.Constant) and int(n.args.exprs[1].value,0)==0 and isinstance(n.args.exprs[2],c_ast.Constant) and int(n.args.exprs[2].value,0)==0x0123;return 'bc_byte_perm('+','.join(args)+')'
   return q+'('+','.join(args)+')'
  if isinstance(n,c_ast.ArrayRef)and isinstance(n.name,c_ast.Cast):
   cast=n.name;assert ctype(cast.to_type.type)==('ptr','UInt32')
   if isinstance(cast.expr,c_ast.ID):base=ex(cast.expr);offset='0'
   else:
    assert isinstance(cast.expr,c_ast.BinaryOp)and cast.expr.op=='+';base=ex(cast.expr.left);offset=ex(cast.expr.right)
   return 'bc_load_u32_le('+base+',Int('+offset+')+4*Int('+ex(n.subscript)+'))'
  if isinstance(n,c_ast.BinaryOp)and n.op=='^':return conv(typex(n),'('+ex(n.left)+'⊻'+ex(n.right)+')')
  return oldex(n)
 env.update(typex=typex,ex=ex,conv=conv,promote=promote)
 lines=[];loops=[]
 def emit(s,l):lines.append(' '*l+s)
 def stmt(n,l):
  if n is None:return
  if isinstance(n,c_ast.Compound):
   for z in n.block_items or[]:stmt(z,l)
  elif isinstance(n,c_ast.Decl):
   t=env['types'][n.name]
   if isinstance(t,tuple):
    assert t[0]=='array';vals=[ex(z)for z in n.init.exprs]if isinstance(n.init,c_ast.InitList)else[t[1]+'(0)']*t[2]
    emit('local '+n.name+'=MVector{'+str(t[2])+','+t[1]+'}(('+','.join(vals)+',))',l)
   else:emit('local '+n.name+'::'+t+' = '+(conv(t,ex(n.init),typex(n.init))if n.init else t+'(0)'),l)
  elif isinstance(n,c_ast.DeclList):
   for z in n.decls:stmt(z,l)
  elif isinstance(n,c_ast.If):
   emit('if '+env['condition'](n.cond),l);stmt(n.iftrue,l+1)
   if n.iffalse:emit('else',l);stmt(n.iffalse,l+1)
   emit('end',l)
  elif isinstance(n,c_ast.For):stmt(n.init,l);emit('while '+env['condition'](n.cond),l);loops.append(n.next);stmt(n.stmt,l+1);loops.pop();stmt(n.next,l+1);emit('end',l)
  elif isinstance(n,c_ast.Return):emit('return '+(ex(n.expr)if n.expr else'nothing'),l)
  elif isinstance(n,c_ast.FuncCall)and n.name.name=='encrypt':
   args=n.args.exprs;assert len(args)==16;targets=[]
   for z in args[-4:]:assert isinstance(z,c_ast.UnaryOp)and z.op=='&';targets.append(ex(z.expr))
   emit('('+','.join(targets)+')=encrypt('+','.join(ex(z)for z in args[:-4])+')',l)
  elif isinstance(n,(c_ast.Assignment,c_ast.UnaryOp,c_ast.FuncCall)):emit(ex(n),l)
  elif isinstance(n,c_ast.EmptyStatement):pass
  else:raise RuntimeError(('stmt',type(n).__name__))
 params=[q.name for q in fn.decl.type.args.params];suffix=['bidx::UInt32','bdimx::UInt32','tidx::UInt32','gdimx::UInt32']if name in ['kernel_w_block','decrypt_vmk_with_mac']else[]
 emit('@inline function '+name+'('+','.join(params+suffix)+')',0);emit('@inbounds begin',1)
 for q in outputargs:emit('local '+q+'::UInt32=0',2)
 stmt(fn.body,2)
 if outputargs:emit('return ('+','.join(outputargs)+')',2)
 emit('end',1);emit('end',0);outputs.append('\n'.join(lines));reports.append({'function':name,'locals':env['types'],'original_output_pointer_tuple_rewrite':outputargs,'alpha_renaming':env['alpha_map']})
header='''# Full source-unrolled SHA256/AES bodies translated by typed C AST.
# GPL source notices/archives retained; UInt32 C wrapping explicit.
using StaticArrays
@inline bc_byte_perm(a,b,s)=bswap(a) # source uses only selector0x0123, b=0
@inline function bc_load_u32_le(a,o)
 @inbounds UInt32(a[o+1])|(UInt32(a[o+2])<<8)|(UInt32(a[o+3])<<16)|(UInt32(a[o+4])<<24)
end
'''
(P/'crypto_device.jl').write_text(header+'\n\n'.join(outputs)+'\n')
(P/'translation_generation_review.json').write_text(json.dumps({'pycparser_version':__import__('pycparser').__version__,'source_files':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in [O/'sha256.h',O/'aes.h',O/'attack.cu',O/'w_blocks.cu']],'typed_emitter_reuse_origin':'STSG/WSM5 previousrecordedhelper, marginalorderedreuse notcoldstart','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha256':hashlib.sha256((P/'typed_emitter_core.py').read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256((P/'crypto_device.jl').read_bytes()).hexdigest(),'functions':reports,'lowering_covariates':['SourceinlinePTXLOP3 uses exact portableBooleanequivalents, compiledinstructionidentitynotclaimed.','AESreadonlytables becomeimmutabletuples, dynamiclookup/compilerresourcebehaviornotassumedsourceconstantmemoryequivalent.','Bytepermselector0123 becomesexactbswap; nohardwareinstructionpromise.','All1Miterations andoriginalunrolledrounds retained; generatedsourceunverifieduntilCPU+GPUchecks.']},indent=2)+'\n')
print('Generated',len(outputs),'parts',len((P/'crypto_device.jl').read_bytes()),'bytes')
