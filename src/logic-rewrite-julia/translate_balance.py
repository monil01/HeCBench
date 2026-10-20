#!/usr/bin/env python3
"""Strict source-driven C++ AST -> Julia rewrite/cut helper candidate generator.
Generated candidates require actual device and full-pipeline verification.
Unimplemented AST constructs fail explicitly; no native-kernel fallback exists.
"""
import argparse,pathlib,sys,re,ctypes,ast,json
p=argparse.ArgumentParser();p.add_argument('--bindings',required=True);p.add_argument('--libclang',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();sys.path.insert(0,a.bindings)
from clang import cindex as C
C.Config.set_library_file(a.libclang);K=C.CursorKind;TK=C.TypeKind
lib=C.conf.lib
for name in ['clang_getCursorBinaryOperatorKind','clang_getCursorUnaryOperatorKind']:
 f=getattr(lib,name);f.argtypes=[C.Cursor];f.restype=ctypes.c_int
for name in ['clang_getBinaryOperatorKindSpelling','clang_getUnaryOperatorKindSpelling']:
 f=getattr(lib,name);f.argtypes=[ctypes.c_int];f.restype=C._CXString;f.errcheck=C._CXString.from_result
lib.clang_Cursor_Evaluate.argtypes=[C.Cursor];lib.clang_Cursor_Evaluate.restype=ctypes.c_void_p
for name in ['clang_EvalResult_getAsLongLong','clang_EvalResult_getAsUnsigned']:
 f=getattr(lib,name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_longlong if name.endswith('LongLong') else ctypes.c_ulonglong
lib.clang_EvalResult_dispose.argtypes=[ctypes.c_void_p]
B=pathlib.Path(__file__).resolve().parent;R=B/'reference_history';raw=(R/'balance.cu').read_text();raw=raw[:raw.index('std::tuple<int *')];pre=next(n.value.value for n in ast.parse((B/'source_audit/inventory.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='pre' for t in n.targets))
pre+='\nunsigned __activemask();\ntemplate<class T>T __shfl_xor_sync(unsigned,T,int);\n'
for host_name in ('gatherByMask','getReadyCovers'):
 match=re.search(r'\bint\s+'+host_name+r'\s*\([^)]*\)\s*\{',raw,re.S)
 if match:
  i=match.end();depth=1
  while depth:depth+=(raw[i]=='{')-(raw[i]=='}');i+=1
  raw=raw[:match.end()]+'\n'*raw[match.end():i-1].count('\n')+raw[i-1:]
name=str(R/'balance.candidate_ast.cpp');tu=C.Index.create().parse(name,args=['-x','c++','-std=c++17','-I'+str(R),'-I/usr/include/c++/12','-I/usr/include/x86_64-linux-gnu/c++/12','-isystem','/usr/lib/gcc/x86_64-linux-gnu/12/include','-ferror-limit=0'],unsaved_files=[(name,pre+re.sub(r'<<<.*?>>>','',raw,flags=re.S))]+[(str(h),re.sub(r'<<<.*?>>>','',h.read_text().replace('#include <cub/cub.cuh>',''),flags=re.S)) for h in R.iterdir() if h.suffix in ('.h','.cuh')])
errors=[str(d) for d in tu.diagnostics if d.severity>=3]
if errors:raise RuntimeError(errors)
SELECT={'recordReconstructedCovers', 'gatherByScannedMask', 'parseOutputRes', 'prepareDataToInsert', 'genReadyMask', 'coverFindingToTable', 'getCoverToNodeIdMapping', 'localCoverTravToTable', 'processPO', 'markIsCoverOutput', 'findLevelNodeRanges', 'checkCoverTravEnd', 'gatherWithFilter', 'addBackLocalArrays', 'retrieveAndNode', 'sharedNodeDrivenPermute'}
FUN={c.spelling:c for c in tu.cursor.get_children() if c.kind==K.FUNCTION_DECL and c.is_definition() and c.spelling in SELECT}
assert set(FUN)==SELECT
TYPES={TK.BOOL:'Bool',TK.CHAR_S:'Int8',TK.SCHAR:'Int8',TK.UCHAR:'UInt8',TK.SHORT:'Int16',TK.USHORT:'UInt16',TK.INT:'Int32',TK.UINT:'UInt32',TK.LONG:'Int64',TK.ULONG:'UInt64',TK.LONGLONG:'Int64',TK.ULONGLONG:'UInt64',TK.FLOAT:'Float32',TK.DOUBLE:'Float64',TK.VOID:'Nothing'}
def ct(t):
 t=t.get_canonical()
 if t.kind in TYPES:return TYPES[t.kind]
 if t.kind==TK.POINTER:return 'LRPtr{'+ct(t.get_pointee())+'}'
 if t.kind==TK.CONSTANTARRAY:return 'LRArray{'+ct(t.element_type)+','+str(t.element_count)+'}'
 if t.kind==TK.RECORD:return {'rw::Cut':'LRCut','rw::TableNode':'LRTableNode','rw::Library':'LRLibrary'}[t.spelling.replace('const ','')]
 raise TypeError(('ctype',t.kind,t.spelling))
def cast(t,s):
 typ=ct(t)
 if typ=='Nothing':return '('+s+'; nothing)'
 if typ.startswith('LRPtr{'):return 'lr_pointer_cast('+typ[6:-1]+','+s+')'
 if typ.startswith(('LRCut','LRTableNode','LRLibrary','LRArray')):return s
 return 'lr_cast('+typ+','+s+')'
def op(n):
 if n.kind in (K.BINARY_OPERATOR,K.COMPOUND_ASSIGNMENT_OPERATOR):return lib.clang_getBinaryOperatorKindSpelling(lib.clang_getCursorBinaryOperatorKind(n))
 return lib.clang_getUnaryOperatorKindSpelling(lib.clang_getCursorUnaryOperatorKind(n))
def kids(n):return list(n.get_children())
class Gen:
 def __init__(self,fn):
  self.fn=fn;self.locals={};self.types={};self.cells=set();self.allocs={};self.allocindex=0
  self.params=list(fn.get_arguments())
  for c in fn.walk_preorder():
   if c.kind in (K.VAR_DECL,K.PARM_DECL):
    key=c.get_usr();self.locals[key]=c.spelling if c.kind==K.PARM_DECL else c.spelling+'_L'+str(c.location.line);self.types[key]=c.type
   if c.kind==K.UNARY_OPERATOR and op(c)=='&':
    z=kids(c)[0]
    while z.kind in (K.UNEXPOSED_EXPR,K.PAREN_EXPR):z=kids(z)[0]
    if z.kind==K.DECL_REF_EXPR:self.cells.add(z.referenced.get_usr())
  for c in fn.walk_preorder():
   if c.kind==K.VAR_DECL:
    t=c.type.get_canonical();key=c.get_usr()
    if t.kind in (TK.CONSTANTARRAY,TK.RECORD)or key in self.cells:self.allocs[key]=ct(t)
  for c in self.params:
   if c.type.get_canonical().kind==TK.RECORD:self.allocs[c.get_usr()]=ct(c.type)
 def local(self,n):return self.locals.get(n.referenced.get_usr(),n.spelling)
 def exp(self,n):
  ch=kids(n);k=n.kind
  if k in (K.UNEXPOSED_EXPR,K.PAREN_EXPR):
   assert len(ch)==1,(k,n.spelling,len(ch));return cast(n.type,self.exp(ch[0])) if k==K.UNEXPOSED_EXPR else '('+self.exp(ch[0])+')'
  if k in (K.INTEGER_LITERAL,K.CXX_BOOL_LITERAL_EXPR,K.CHARACTER_LITERAL):
   e=lib.clang_Cursor_Evaluate(n)
   if not e:raise TypeError(('literal',n.spelling,[x.spelling for x in n.get_tokens()]))
   v=lib.clang_EvalResult_getAsUnsigned(e) if ct(n.type).startswith('UInt') else lib.clang_EvalResult_getAsLongLong(e);lib.clang_EvalResult_dispose(e)
   return cast(n.type,str(v))
  if k==K.DECL_REF_EXPR:
   if n.referenced.get_usr() in self.cells:return self.local(n)+'[0]'
   if n.spelling in ('P','N','GPUexpected','replaceHasFullCorrsp'):return 'ctx.'+n.spelling+'[0]'
   if n.spelling=='HASHTABLE_EMPTY_VALUE':return 'typemax('+ct(n.type)+')'
   if n.referenced.get_usr() not in self.locals and n.type.get_canonical().kind in TYPES:
    ev=lib.clang_Cursor_Evaluate(n)
    if ev:
     value=lib.clang_EvalResult_getAsLongLong(ev);lib.clang_EvalResult_dispose(ev);return cast(n.type,str(value))
   return self.local(n)
  if k==K.MEMBER_REF_EXPR:
   b=ch[0]
   if b.spelling in ('blockIdx','blockDim','threadIdx','gridDim'):
    s={'blockIdx':'(Int32(blockIdx().'+n.spelling+')-Int32(1))','threadIdx':'(Int32(threadIdx().'+n.spelling+')-Int32(1))','blockDim':'Int32(blockDim().'+n.spelling+')','gridDim':'Int32(gridDim().'+n.spelling+')'}[b.spelling];return s
   return '('+self.exp(b)+').'+n.spelling
  if k==K.ARRAY_SUBSCRIPT_EXPR:return '('+self.exp(ch[0])+')['+self.exp(ch[1])+']'
  if k in (K.BINARY_OPERATOR,K.COMPOUND_ASSIGNMENT_OPERATOR):
   x,y=ch;oper=op(n)
   if oper in ('=','+=','-=','*=','|=','&=','^=','<<=','>>=','%=','/='):
    lhs=self.exp(x);rhs=self.exp(y)
    if oper!='=':rhs=self.binary(oper[:-1],lhs,rhs,x.type)
    rhs=cast(x.type,rhs)
    return '(_lr_rhs='+rhs+'; '+lhs+' = _lr_rhs; _lr_rhs)'
   if oper==',':return '('+self.exp(x)+'; '+self.exp(y)+')'
   return self.binary(oper,self.exp(x),self.exp(y),n.type)
  if k==K.UNARY_OPERATOR:
   x=ch[0];oper=op(n)
   if oper=='&':return self.address(x)
   if oper=='*':return '('+self.exp(x)+')[0]'
   if oper in ('++','--'):
    lhs=self.exp(x);delta='+'if oper=='++'else'-';new=cast(x.type,'('+lhs+delta+'Int32(1))')
    # Clang unary opcode distinguishes post/pre only by source token placement.
    post=[t.spelling for t in n.get_tokens()][-1:]==[oper]
    if post:return '(_lr_old='+lhs+'; '+lhs+'='+new+'; _lr_old)'
    return '('+lhs+'='+new+'; '+lhs+')'
   if oper=='!':return '(!lr_truth('+self.exp(x)+'))'
   return cast(n.type,'('+('~'if oper=='~'else oper)+self.exp(x)+')')
  if k==K.CONDITIONAL_OPERATOR:return '(lr_truth('+self.exp(ch[0])+') ? '+self.exp(ch[1])+' : '+self.exp(ch[2])+')'
  if k in (K.CSTYLE_CAST_EXPR,K.CXX_STATIC_CAST_EXPR,K.CXX_FUNCTIONAL_CAST_EXPR):return cast(n.type,self.exp(ch[-1])) if ch else 'nothing'
  if k==K.CALL_EXPR:
   f=n.referenced.spelling if n.referenced else n.spelling
   if f=='__assert_fail':return '(CUDA.@cuassert false)'
   if f=='printf':return 'Int32(0)' # source diagnostics only; no argument sideeffects in selected calls
   args=[self.exp(x) for x in n.get_arguments()]
   if f in ('atomicCAS','atomicAdd','atomicSub'):return 'lr_'+f+'('+','.join(args)+')'
   if f=='Cut' and len(args)==1:return args[0]
   if f=='operator=' and len(args)==2:return 'lr_copy!('+args[0]+','+args[1]+','+ct(list(n.get_arguments())[0].type)+')'
   if f=='__assert_fail':return '(CUDA.@cuassert false)'
   if f=='printf':return 'Int32(0) # omitted source diagnostic-only device printf\n'
   if f in ('AigIsNode','AigIsPIConst','AigNodeNot','AigNodeID','AigNodeIsComplement','AigNodeLitCond','AigNodeNotCond','formAndNodeKey','checkTrivialAndCases','unbindAndNodeKeys','insert_single_no_update','retrieve_single'):return 'lr_s_'+f+'('+','.join(args)+')'
   if f in ('max','min'):return f+'('+','.join(args)+')'
   if f in SELECT:return 'lr_'+f+'(ctx'+(' ,'+','.join(args)if args else'')+')'
   raise TypeError(('call',f,[(z.kind.name,z.spelling,z.type.spelling)for z in n.get_arguments()],[(z.kind.name,z.spelling,z.type.spelling)for z in ch]))
  raise TypeError(('expression',k,n.spelling,[t.spelling for t in n.get_tokens()][:12]))
 def binary(self,oper,x,y,t):
  if oper=='^':return cast(t,'xor('+x+','+y+')')
  if oper in ('&&','||'):return '(lr_truth('+x+') '+oper+' lr_truth('+y+'))'
  if oper=='/':return cast(t,'div('+x+','+y+')')
  if oper=='%':return cast(t,'rem('+x+','+y+')')
  return cast(t,'('+x+' '+oper+' '+y+')')
 def address(self,n):
  ch=kids(n)
  if n.kind in (K.UNEXPOSED_EXPR,K.PAREN_EXPR):return self.address(ch[0])
  if n.kind==K.DECL_REF_EXPR:
   if n.spelling in ('P','N','GPUexpected','replaceHasFullCorrsp'):return 'ctx.'+n.spelling
   return self.local(n)
  if n.kind==K.MEMBER_REF_EXPR:return 'lr_field_address('+self.exp(ch[0])+',Val(:'+n.spelling+'))'
  if n.kind==K.ARRAY_SUBSCRIPT_EXPR:return '('+self.exp(ch[0])+' + '+self.exp(ch[1])+')'
  raise TypeError(('address',n.kind,n.spelling))
 def cleanup(self):return '; '.join('lr_free(Ptr{Cvoid}(getfield(_lr_mem_'+str(i)+',:base)))' for i,_ in enumerate(self.allocs))
 def statement(self,n,level=1):
  ch=kids(n);k=n.kind;pad='    '*level
  if k==K.COMPOUND_STMT:return '\n'.join(self.statement(c,level) for c in ch)
  if k==K.DECL_STMT:return '\n'.join(self.statement(c,level)for c in ch)
  if k==K.VAR_DECL:
   key=n.get_usr();var=self.locals[key];t=n.type.get_canonical();init=ch[-1]if ch and ch[-1].kind not in (K.TYPE_REF,) else None
   if t.kind==TK.CONSTANTARRAY:
    if init and init.kind==K.INIT_LIST_EXPR:
     vals=[]
     def flatten(c):
      if c.kind==K.INIT_LIST_EXPR:
       for z in kids(c):flatten(z)
      else:vals.append(self.exp(c))
     flatten(init);base=t
     while base.kind==TK.CONSTANTARRAY:base=base.element_type
     return '\n'.join(pad+'LRPtr{'+ct(base)+'}(getfield('+var+',:base))['+str(i)+']='+v for i,v in enumerate(vals))
    return ''
   if init is None:return pad+'local '+var if key not in self.allocs else ''
   rhs=self.exp(init)
   if t.kind==TK.RECORD:return pad+'lr_copy!('+var+','+rhs+','+ct(t)+')'
   if key in self.cells:return pad+var+'[0]='+cast(t,rhs)
   return pad+var+'='+cast(t,rhs)
  if k==K.IF_STMT:
   out=pad+'if lr_truth('+self.exp(ch[0])+')\n'+self.statement(ch[1],level+1)
   if len(ch)==3:out+='\n'+pad+'else\n'+self.statement(ch[2],level+1)
   return out+'\n'+pad+'end'
  if k==K.FOR_STMT:
   assert len(ch)==4,('for',len(ch))
   init,cond,inc,body=ch;out=self.statement(init,level)+'\n'+pad+'while lr_truth('+self.exp(cond)+')\n'
   # Continue must execute C for-loop increment before continuing.
   previous=getattr(self,'continue_inc',None);self.continue_inc=inc
   out+=self.statement(body,level+1)+'\n'+'    '*(level+1)+self.exp(inc)+'\n'+pad+'end';self.continue_inc=previous;return out
  if k==K.WHILE_STMT:
   old=getattr(self,'continue_inc',None);self.continue_inc=None
   out=pad+'while lr_truth('+self.exp(ch[0])+')\n'+self.statement(ch[1],level+1)+'\n'+pad+'end';self.continue_inc=old;return out
  if k==K.RETURN_STMT:
   value=self.exp(ch[0])if ch else'nothing';return pad+'begin _lr_result='+cast(self.fn.result_type,value)+'; '+self.cleanup()+'; return _lr_result end'
  if k==K.CONTINUE_STMT:
   inc=getattr(self,'continue_inc',None);return pad+(self.exp(inc)+'; 'if inc else'')+'continue'
  if k==K.BREAK_STMT:return pad+'break'
  if k==K.NULL_STMT:return ''
  return pad+self.exp(n)
 def generate(self):
  lines=['# Original '+self.fn.spelling+' source device body; candidate, not certified.','@inline function lr_'+self.fn.spelling+'(ctx'+''.join(','+c.spelling for c in self.params)+')'+('::'+ct(self.fn.result_type) if self.fn.result_type.get_canonical().kind in TYPES else'')]
  for i,(key,typ) in enumerate(self.allocs.items()):
   var=self.locals[key];lines.append('    _lr_mem_'+str(i)+'=lr_alloc('+typ+')')
   if any(c.get_usr()==key for c in self.params):lines.append('    lr_copy!(_lr_mem_'+str(i)+','+var+','+typ+')')
   lines.append('    '+var+'='+('_lr_mem_'+str(i)+'[0]'if typ in ('LRCut','LRTableNode','LRLibrary')else 'lr_pointer_cast('+typ[8:-1].rsplit(',',1)[0]+',_lr_mem_'+str(i)+')'if typ.startswith('LRArray{')else'_lr_mem_'+str(i)))
  body=next(c for c in kids(self.fn)if c.kind==K.COMPOUND_STMT);lines.append(self.statement(body));lines.append('    '+self.cleanup());lines.append('    return nothing\nend')
  return '\n'.join(lines)
text='# GENERATED from original source; full-pipeline verification PENDING.\n'+ '\n\n'.join(Gen(FUN[n]).generate()for n in FUN)+'\n'
a.output.write_text(text);print('Generated',len(FUN),'source function candidates, lines',len(text.splitlines()))
