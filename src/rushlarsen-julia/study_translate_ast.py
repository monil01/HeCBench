from pathlib import Path
import sys,re,json,hashlib
sys.path.insert(0,'/tmp/wsm5-python-tools')
from pycparser import c_parser,c_ast
p=Path(__file__).resolve().parent;r=p.parent.parent;s=(r/'src/rushlarsen-cuda/kernels.cu').read_text();start=s.index('    // Assign states');body=s[start:s.rfind('  }')];body=re.sub(r'//[^\n]*','',body);ast=c_parser.CParser().parse('void body(double* states,double t,double dt,const double* parameters,int n,int i){'+body+'}')
def ex(n):
 if isinstance(n,c_ast.ID):return n.name
 if isinstance(n,c_ast.Constant):return repr(float(n.value))if n.type in ['double','float']else 'Int32('+str(int(n.value,0))+')'
 if isinstance(n,c_ast.ArrayRef):return ex(n.name)+'[('+ex(n.subscript)+')+Int32(1)]'
 if isinstance(n,c_ast.BinaryOp):return '('+ex(n.left)+' '+n.op+' '+ex(n.right)+')'
 if isinstance(n,c_ast.UnaryOp):return '('+n.op+ex(n.expr)+')'
 if isinstance(n,c_ast.TernaryOp):return '('+ex(n.cond)+' ? '+ex(n.iftrue)+' : '+ex(n.iffalse)+')'
 if isinstance(n,c_ast.FuncCall):
  name=ex(n.name);args=[ex(x)for x in n.args.exprs]
  if name=='fabs':name='abs'
  if name=='pow':return '('+args[0]+' ^ '+args[1]+')'
  assert name in ['abs','exp','log','sqrt','floor'],name
  return name+'('+','.join(args)+')'
 raise TypeError(type(n))
lines=['# Full original474line C AST translation; no scientific correction.','# Every source state/parameter/rate is Float64. Original SoA/old-state caches retained.']
h=(r/'src/rushlarsen-cuda/utils.h').read_text()
for prefix in ['STATE','PARAM']:
 for i,name in enumerate(re.findall(r'^  '+prefix+r'_(\w+),',h,re.M)):lines.append('const '+prefix+'_'+name+'=Int32('+str(i)+')')
lines+=['function rush_body!(states,t::Float64,dt::Float64,parameters,n::Int32,i::Int32)',' if i<n','  @inbounds begin'];names=[]
for n in ast.ext[0].body.block_items:
 if isinstance(n,c_ast.Decl):
  assert n.type.type.names==['double'];assert n.name not in names;names.append(n.name);lines.append('   local '+n.name+'::Float64 = '+ex(n.init))
 elif isinstance(n,c_ast.Assignment):lines.append('   '+ex(n.lvalue)+' = '+ex(n.rvalue))
 else:raise TypeError(type(n))
lines+=['  end',' end',' return nothing','end'];(p/'study_device.jl').write_text('\n'.join(lines)+'\n');(p/'translation_generation_review.json').write_text(json.dumps({'generator':'pycparser3.11 full originalCstatement/expressionAST, explicitFloat64declarations andInt32enum/indexarithmetic','native_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'all_source_scalar_declarations':len(names),'state_store_count':sum(isinstance(x,c_ast.Assignment)for x in ast.ext[0].body.block_items),'source_expression_parentheses':'ASTpreserved; nohandrewritingoflongJacobianexpressions','algorithm_candidates':1,'independent_generation':False,'reuse':'PriorWSM5experience+installedpycparser tool reused; newcompactgeneratorwritteninCUDAjltranslationwindow'},indent=2)+'\n');print('generated',len(lines),'lines',len(names),'scalarDoublebindings')
