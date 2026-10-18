from pathlib import Path
import re,json,subprocess,sys,hashlib,shutil
sys.path.insert(0,'/tmp/portable-c-parser')
from pycparser import c_parser,c_ast as C
R=Path('/home/imo/HeCBench_julia');P=R/'src/segsort-julia';N=R/'src/segsort-cuda/src'
(P/'reference_original').mkdir(exist_ok=True)
for n in ['main.jl','Makefile']:
 if not(P/'reference_original'/n).exists():shutil.copy2(P/n,P/'reference_original'/n)
for suffix in ['cuda.jl','KA','JACC']:(P/f'segsort-julia-{suffix}').mkdir(exist_ok=True)
FILES=['bb_comput_common.cuh','bb_exch_keys.cuh','bb_exch.cuh','bb_bin.cuh','bb_comput_s_keys.cuh','bb_comput_s.cuh','bb_comput_l_keys.cuh','bb_comput_l.cuh']
funcs=[];macros=[];refs={};rawdefs=[]
for fname in FILES:
 raw=(N/fname).read_text();s=re.sub(r'/\*.*?\*/','',raw,flags=re.S);s=re.sub(r'//[^\n]*','',s)
 macros += [line for line in s.splitlines()if re.match(r'\s*#define\s+(CMP_SWP|EQL_SWP|SWP)',line)]
 # Restrict to actual __device__/__global__ definitions; host dispatch separate.
 for match in re.finditer(r'__(device|global)__\s+(?:inline\s+)?(\w+)\s+(\w+)\s*\((.*?)\)\s*\{',s,re.S):
  kind,ret,name,args=match.groups();pos=match.end();depth=1;j=pos
  while depth:
   depth+=(s[j]=='{')-(s[j]=='}');j+=1
  body=s[pos:j-1];arglist=[a.strip()for a in args.split(',')];refnames=[re.search(r'(\w+)\s*$',a)[1]for a in arglist if'&'in a];key=(name,len(arglist));refs[key]=refnames
  args2=args.replace('&','');body=re.sub(r'std::numeric_limits<K>::max\(\)','2147483647',body)
  body=body.replace('__shared__','').replace('__syncthreads()','sg_barrier()')
  body=re.sub(r'^\s*#.*$','',body,flags=re.M)
  text=f'{ret} {name}({args2})'+'{\n'+body+'\n}\n'
  rawdefs.append({'source':fname,'source_sha256':hashlib.sha256((N/fname).read_bytes()).hexdigest(),'name':name,'argc':len(arglist),'kernel':kind=='global','refs':refnames,'code':text})
joined='#define SEGBIN_NUM 13\ntypedef int K;typedef unsigned long long T;typedef int Offset;typedef int bool;\n'+'\n'.join(macros)+'\n'+'\n'.join(d['code']for d in rawdefs)
tmp=Path('/tmp/segsort_ast_input.c');tmp.write_text(joined)
pre=subprocess.check_output(['gcc','-E','-P','-x','c',str(tmp)],text=True)
Path('/tmp/segsort_ast_preprocessed.c').write_text(pre)
ast=c_parser.CParser().parse(pre)
defs=[n for n in ast.ext if isinstance(n,C.FuncDef)];assert len(defs)==len(rawdefs)
known={x['name']for x in rawdefs}
def V(name):return 'v_'+name
def typename(t,fn):
 while isinstance(t,(C.TypeDecl,C.PtrDecl,C.ArrayDecl)):t=t.type
 if isinstance(t,C.IdentifierType):
  name=' '.join(t.names)
  return {'int':'Int32','K':'Int32','Offset':'Int32','T':'Int32'if fn=='warp_exclusive_sum'else'UInt64','bool':'Bool','unsigned':'UInt32','unsigned int':'UInt32','unsigned long long':'UInt64','void':'Nothing'}[name]
 raise RuntimeError(('type',t))
class Translator:
 def __init__(self,node,meta):
  self.node=node;self.meta=meta;self.name=meta['name'];self.locals={};self.params=node.decl.type.args.params
  self.paramnames={p.name for p in self.params};self.pointers={p.name for p in self.params if isinstance(p.type,C.PtrDecl)}
  class Visitor(C.NodeVisitor):
   def visit_Decl(visitor,d):
    if d.name not in self.paramnames:self.locals[d.name]=d
    visitor.generic_visit(d)
  Visitor().visit(node.body)
 def expr(self,x):
  if x is None:return ''
  if isinstance(x,C.ID):
   if x.name=='true':return 'true'
   if x.name=='false':return 'false'
   return V(x.name)
  if isinstance(x,C.Constant):
   a=re.sub(r'[uUlL]+$','',x.value)
   if x.type in ['int','unsigned int','long int','unsigned long int','unsigned long long int']:
    n=int(a,0);return f'{"UInt32"if n>2147483647 else"Int32"}({a})'
   raise RuntimeError(('const',x.type,x.value))
  if isinstance(x,C.BinaryOp):
   a,b=self.expr(x.left),self.expr(x.right)
   if x.op=='/':return f'div({a},{b})'
   if x.op=='^':return f'xor({a},{b})'
   if x.op=='&&':return f'({self.cond(x.left)} && {self.cond(x.right)})'
   if x.op=='||':return f'({self.cond(x.left)} || {self.cond(x.right)})'
   return f'({a} {x.op} {b})'
  if isinstance(x,C.UnaryOp):
   if x.op=='&':
    assert isinstance(x.expr,C.ArrayRef),x
    return f'sg_pointer({self.expr(x.expr.name)},{self.expr(x.expr.subscript)})'
   if x.op=='!':return f'!({self.cond(x.expr)})'
   if x.op in ['p++','++','p--','--']:return f'{self.expr(x.expr)} {"+"if"+"in x.op else"-"}= Int32(1)'
   if x.op in ['-','+','~']:return f'({x.op}{self.expr(x.expr)})'
   raise RuntimeError(('unary',x.op))
  if isinstance(x,C.ArrayRef):return f'{self.expr(x.name)}[{self.expr(x.subscript)}]'
  if isinstance(x,C.StructRef):
   obj=x.name.name;field=x.field.name
   return {'threadIdx.x':'ctx[1]','blockIdx.x':'ctx[2]','blockIdx.y':'ctx[3]','blockDim.x':'ctx[4]'}[obj+'.'+field]
  if isinstance(x,C.TernaryOp):return f'({self.cond(x.cond)} ? {self.expr(x.iftrue)} : {self.expr(x.iffalse)})'
  if isinstance(x,C.Cast):return f'{typename(x.to_type.type,self.name)}({self.expr(x.expr)})'
  if isinstance(x,C.Assignment):return f'{self.expr(x.lvalue)} {x.op} {self.expr(x.rvalue)}'
  if isinstance(x,C.FuncCall):
   name=x.name.name;args=x.args.exprs if x.args else[];ex=[self.expr(a)for a in args]
   if name in known:return f'sg_{name}(ctx'+(' ,'+','.join(ex)if ex else'')+')'
   mapping={'__shfl_sync':'CUDA.shfl_sync','__shfl_xor_sync':'CUDA.shfl_xor_sync','__shfl_up_sync':'CUDA.shfl_up_sync','__shfl_down_sync':'CUDA.shfl_down_sync','sg_barrier':'CUDA.sync_threads','atomicAdd':'CUDA.atomic_add!','atomicMax':'CUDA.atomic_max!','min':'min','max':'max'}
   if name not in mapping:raise RuntimeError(('call',name,self.name))
   return mapping[name]+'('+','.join(ex)+')'
  if isinstance(x,C.ExprList):return ';'.join(self.expr(a)for a in x.exprs)
  raise RuntimeError(('expr',type(x).__name__,self.name))
 def cond(self,x):
  if isinstance(x,C.BinaryOp)and x.op in ['<','>','<=','>=','==','!=','&&','||']:return self.expr(x)
  if isinstance(x,C.UnaryOp)and x.op=='!':return self.expr(x)
  return f'({self.expr(x)} != 0)'
 def stmt(self,x,indent=' '):
  if x is None:return []
  if isinstance(x,C.Compound):
   return [line for n in (x.block_items or[])for line in self.stmt(n,indent)]
  if isinstance(x,C.Decl):
   if isinstance(x.type,C.ArrayDecl):return [] # static shared allocation hoisted, matching static CUDA storage.
   return [indent+V(x.name)+' = '+self.expr(x.init)]if x.init is not None else[]
  if isinstance(x,C.DeclList):return [v for n in x.decls for v in self.stmt(n,indent)]
  if isinstance(x,C.If):
   lines=[indent+'if '+self.cond(x.cond)]+self.stmt(x.iftrue,indent+' ')
   if x.iffalse:lines +=[indent+'else']+self.stmt(x.iffalse,indent+' ')
   return lines+[indent+'end']
  if isinstance(x,C.For):
   return self.stmt(x.init,indent)+[indent+'while '+self.cond(x.cond)]+self.stmt(x.stmt,indent+' ')+self.stmt(x.next,indent+' ')+[indent+'end']
  if isinstance(x,C.While):return [indent+'while '+self.cond(x.cond)]+self.stmt(x.stmt,indent+' ')+[indent+'end']
  if isinstance(x,C.Return):return [indent+'return '+self.expr(x.expr)]
  if isinstance(x,C.Break):return [indent+'break']
  if isinstance(x,C.EmptyStatement):return []
  if isinstance(x,C.FuncCall):
   ref=refs.get((x.name.name,len(x.args.exprs)if x.args else 0),[])
   if ref:
    meta=next(d for d in rawdefs if d['name']==x.name.name and d['argc']==len(x.args.exprs));argnames=[p.name for p in next(n for n in defs if n.decl.name==x.name.name and len(n.decl.type.args.params)==len(x.args.exprs)).decl.type.args.params];indices=[argnames.index(n)for n in ref]
    dest=[self.expr(x.args.exprs[i])for i in indices]
    return [indent+','.join(dest)+(' ,'if len(dest)==1 else'')+' = '+self.expr(x)]
  if isinstance(x,(C.Assignment,C.UnaryOp,C.FuncCall,C.ExprList)):return [indent+self.expr(x)]
  raise RuntimeError(('stmt',type(x).__name__,self.name))
 def generate(self):
  args=','.join(V(p.name)for p in self.params);lines=[f'# {self.meta["source"]} {self.meta["source_sha256"]}',f'@inline function sg_{self.name}(ctx,{args})',' @inbounds begin']
  for name,d in self.locals.items():
   if isinstance(d.type,C.ArrayDecl):
    typ=typename(d.type,self.name);lines.append(f' {V(name)}=CSpan(CUDA.CuStaticSharedArray({typ},{self.expr(d.type.dim)}),Int32(0))')
   elif isinstance(d.type,C.PtrDecl):lines.append(f' local {V(name)}') # assigned typed span before source use.
   else:
    typ=typename(d.type,self.name);lines.append(f' local {V(name)}::{typ} = zero({typ})')
  lines+=self.stmt(self.node.body)
  lines.append(' end')
  if self.meta['refs']:lines.append(' return ('+','.join(V(n)for n in self.meta['refs'])+',)')
  elif typename(self.node.decl.type.type,self.name)=='Nothing':lines.append(' nothing')
  lines.append('end')
  return '\n'.join(lines)
preamble=r'''
# Mechanically translated generated CUDA networks: C spans retain zero-based indexing/pointer offsets.
struct CSpan{A}
 data::A
 offset::Int32
end
@inline Base.getindex(a::CSpan,i::Integer)=@inbounds a.data[a.offset+i+1]
@inline Base.setindex!(a::CSpan,v,i::Integer)=(@inbounds a.data[a.offset+i+1]=v)
@inline Base.:+(a::CSpan,i::Integer)=CSpan(a.data,a.offset+Int32(i))
@inline sg_pointer(a::CSpan,i::Integer)=pointer(a.data,a.offset+i+1)
'''
out=preamble+'\n'+'\n\n'.join(Translator(node,meta).generate()for node,meta in zip(defs,rawdefs))
(P/'study_generated.jl').write_text(out)
kernelmeta=[]
wrappers=['using CUDA,Printf','include("../study_generated.jl")']
for node,meta in zip(defs,rawdefs):
 if not meta['kernel']:continue
 # Unique entry ID disambiguates keys/pairs arity while source helper overloads retain original dispatch.
 uid=meta['name']+'_'+str(meta['argc']);params=node.decl.type.args.params;names=[p.name for p in params]
 kernelmeta.append({k:v for k,v in meta.items()if k!='code'}|{'uid':uid,'parameter_names':names,'pointer_parameters':[p.name for p in params if isinstance(p.type,C.PtrDecl)]})
 args=','.join('a_'+n for n in names);call=','.join(('CSpan(a_'+p.name+',Int32(0))')if isinstance(p.type,C.PtrDecl)else('a_'+p.name)for p in params)
 wrappers +=[f'function sg_entry_{uid}({args})',' ctx=(Int32(threadIdx().x-1),Int32(blockIdx().x-1),Int32(blockIdx().y-1),Int32(blockDim().x))',f' sg_{meta["name"]}(ctx,{call})',' nothing','end',f'function launch_generated(::Val{{:{uid}}},{args};blocks,threads)',f' @cuda blocks=blocks threads=threads sg_entry_{uid}({args})',' nothing','end']
wrappers+=['include("../study_common.jl")','segsort_study(launch_generated)']
(P/'segsort-julia-cuda.jl/main.jl').write_text('\n'.join(wrappers)+'\n')
(P/'generated_kernel_inventory.json').write_text(json.dumps({'parser':'pycparser2.22tasklocal/tmp/portable-c-parser','sources':[{'path':str((N/f).relative_to(R)),'sha256':hashlib.sha256((N/f).read_bytes()).hexdigest()}for f in FILES],'device_functions':len(rawdefs),'authored_entry_kernels':len(kernelmeta),'kernels':kernelmeta,'mapping':'Originalgeneratedregisterexchanges/sharedpermutations andlongmergepathtranscribed, notdifferentgenericnetwork orlibrarysort. C++scalarrefs becomeexplicitreturnedtuples, sourcearrayszero-basedCSpan, KInt32/OffsetInt32/TUInt64actualbenchmarktypes.','padding':'UninitializedCscalarpaddingregisters initializedzero inJuliatodefineotherwiseundefSSA; allnonpaddingkeys<INT_MAX so dummypermutationpayloadnotobservable; actualfullCPUchecksrequired.','source_scope':'Bothkeys-onlyandUInt64pairs, all13bins/largepipeline; originaldefault2^25 scale separateunmeasureduntilstudy.'},indent=2)+'\n')
# Save task-local parser generator source for reproducibility.
shutil.copy2('/tmp/segsort_translate.py',P/'generate_julia_sources.py')
print('translated devicefunctions',len(rawdefs),'kernels',len(kernelmeta),'bytes',len(out))
