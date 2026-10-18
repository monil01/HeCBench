from pathlib import Path
import re,json,hashlib
r=Path('/home/imo/HeCBench_julia');p=r/'src/sw4ck-julia';d=p/'sw4ck-julia-cuda.jl';d.mkdir(exist_ok=True);(d/'original_main.jl').write_bytes((p/'main.jl').read_bytes()) if not (d/'original_main.jl').exists() else None;(p/'original_Makefile').write_bytes((p/'Makefile').read_bytes());(p/'LICENSE').write_bytes((r/'src/sw4ck-cuda/LICENSE').read_bytes());(p/'main.jl').write_text('include(joinpath(@__DIR__,"sw4ck-julia-cuda.jl","main.jl"))\n')
macros={'mu':('a_mu','base+({0})+ni*({1})+nij*({2})+1'),'la':('a_lambda','base+({0})+ni*({1})+nij*({2})+1'),'jac':('a_jac','base+({0})+ni*({1})+nij*({2})+1'),'u':('a_u','base-nijk+({1})+ni*({2})+nij*({3})+nijk*({0})+1'),'lu':('a_lu','base-nijk+({1})+ni*({2})+nij*({3})+nijk*({0})+1'),'met':('a_met','base-nijk+({1})+ni*({2})+nij*({3})+nijk*({0})+1'),'strx':('a_strx','({0})-ifirst+1'),'stry':('a_stry','({0})-jfirst+1'),'acof':('a_acof','({0})-1+6*(({1})-1)+48*(({2})-1)+1'),'bope':('a_bope','({0})-1+6*(({1})-1)+1'),'ghcof':('a_ghcof','({0})'),'acof_no_gp':('a_acof_no_gp','({0})-1+6*(({1})-1)+48*(({2})-1)+1'),'ghcof_no_gp':('a_ghcof_no_gp','({0})')}
def expand(s):
 names='|'.join(sorted(macros,key=len,reverse=True));pattern=r'\b('+names+r')\s*\('
 while (m:=re.search(pattern,s)):
  st=m.end();depth=1;i=st
  while depth:
   if s[i]=='(':depth+=1
   elif s[i]==')':depth-=1
   i+=1
  args=s[st:i-1].split(',');arr,idx=macros[m.group(1)];s=s[:m.start()]+arr+'['+idx.format(*args)+']'+s[i:]
 return s
allcode=[];manifest=[]
for number in range(1,6):
 path=r/f'src/sw4ck-cuda/kernel{number}.cpp';raw=path.read_text();raw=re.sub(r'//[^\n]*','',raw);head,body=raw.split('{',1);body=body.rsplit('}',1)[0];params=[]
 for arg in head[head.index('(')+1:head.rindex(')')].split(','):
  params.append(re.findall(r'\b[A-Za-z_]\w*\b',arg)[-1])
 # Tokens are statements or brace/parens control constructs; semicolons in for headers kept.
 toks=[];start=0;depth=0
 for i,ch in enumerate(body):
  if ch=='(':depth+=1
  if ch==')':depth-=1
  if depth==0 and ch in '{};':
   stmt=body[start:i].strip()
   if stmt:toks.append(stmt)
   if ch in '{}':toks.append(ch)
   start=i+1
 scopes=[{}];types={};counter=0;out=[];pending=None
 def subst(s):
  env={}
  for scope in scopes:env.update(scope)
  s=re.sub(r'\b[A-Za-z_]\w*\b',lambda m:env.get(m[0],m[0]),s);return expand(s)
 def decl(s):
  nonlocal_dummy=None
  global counter
  kind,rest=s.split(None,1);out=[]
  parts=[];st=0;dep=0
  for j,ch in enumerate(rest):
   if ch=='(':dep+=1
   elif ch==')':dep-=1
   elif ch==',' and dep==0:parts.append(rest[st:j]);st=j+1
  parts.append(rest[st:])
  for part in parts:
   name,sep,value=part.partition('=');name=name.strip();counter+=1;new=f'{name}_s{counter}';scopes[-1][name]=new;types[new]=kind
   if sep:out.append(new+' = '+('Float64' if kind=='float_sw4' else 'Int32')+'('+subst(value.strip())+')')
   else:out.append('local '+new)
  return out
 for tok in toks:
  if tok=='{':
   scopes.append({})
   if pending:
    typ,val=pending;pending=None
    if typ=='for':
     name,lo,hi=val;counter+=1;new=f'{name}_s{counter}';scopes[-1][name]=new;types[new]='int';out.append('for '+new+' in Int32('+subst(lo)+'):Int32('+subst(hi)+')')
    else:out.append('if '+subst(val))
   else:out.append('begin')
  elif tok=='}':out.append('end');scopes.pop()
  elif tok.startswith('if '):pending=('if',tok[tok.index('(')+1:tok.rindex(')')])
  elif tok.startswith('for '):
   m=re.fullmatch(r'for\s*\(int (\w+)\s*=\s*(.*?);\s*\1\s*<=\s*(.*?);\s*\1\+\+\)',tok,re.S);assert m,tok;pending=('for',m.groups())
  elif re.match(r'^(float_sw4|int)\s+',tok):
   if 'threadIdx' in tok:
    name=re.match(r'int (\w+)',tok)[1];axis={'i':'x','j':'y','k':'z'}[name];dim={'i':1,'j':2,'k':3}[name];out.append(f'{name} = Int32(start{dim-1}+(blockIdx().{axis}-1)*blockDim().{axis}+threadIdx().{axis}-1)')
   else:out+=decl(tok)
  else:
   tok=subst(tok)
   if (m:=re.match(r'^(\w+)\s*=\s*(.*)$',tok,re.S)) and types.get(m[1])=='float_sw4':tok=m[1]+' = Float64('+m[2]+')'
   out.append(tok)
 assert len(scopes)==1
 prefix='''ni=ilast-ifirst+1
nij=ni*(jlast-jfirst+1)
nijk=nij*(klast-kfirst+1)
base=-(ifirst+ni*jfirst+nij*kfirst)
i6=1.0/6.0;tf=0.75;c1=2.0/3.0;c2=-1.0/12.0
'''
 code='function kernel'+str(number)+'!('+','.join(params)+')\n'+prefix+'\n'.join(out)+'\nreturn\nend\n';allcode.append(code);manifest.append({'kernel':number,'native_source':str(path),'native_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parameter_names':params,'translation_method':'Source arithmetic expressions and lexical-scoped variables preserved; Carrayaccessmacros expanded to1basedJulia; CPU and framework kernels share generatedexpressionbody, not independentoracle.'})
(p/'study_reference_kernels.jl').write_text('# Derived from SW4CK GPL-2.0-only source; see LICENSE and source_translation_manifest.json.\n'+'\n'.join(allcode));(p/'source_translation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(p/'translate_source.py').write_bytes(Path(__file__).read_bytes());print('translated',len(allcode),'kernels',sum(len(s.splitlines())for s in allcode),'lines')
# CPU C++ source bodies, unchanged arithmetic and macros, only host-call qualifiers/index origins adapted.
cpus=[]
for k in range(1,6):
 s=(r/f'src/sw4ck-cuda/kernel{k}.cpp').read_text().replace('__global__','').replace(f'void kernel{k}(',f'void cpu_kernel{k}(')
 for axis,name,st in [('x','i',0),('y','j',1),('z','k',2)]:s=re.sub(r'int '+name+r' = start'+str(st)+r' \+ threadIdx\.'+axis+r' \+ blockIdx\.'+axis+r' \* blockDim\.'+axis+r';',f'int {name} = start{st};',s)
 cpus.append(s)
(p/'study_cpu_native.h').write_text('// Original GPL-2.0-only kernel arithmetic; host index adapter only.\n'+'\n'.join(cpus))
