# Reproduce from archived exact preprocessor output; running replaces generated bodies and source map.
from pathlib import Path
import re,json,hashlib
p=Path(__file__).resolve().parent
s=(p/'native_expanded_kernels.cpp').read_text();parts=re.split(r'__global__\s*void\s+',s)[1:];functions=[]
for x in parts:
 name=x[:x.index('(')].strip();head=x[:x.index('{')];body=x[x.index('{')+1:x.rindex('}')];functions.append((name,head,body))
assert len(functions)==25

def expr(s,component=False):
 s=s.strip()
 # C AoSfloat2 components become scalar accesses preserving8B physicalstride.
 s=re.sub(r'(sigma_in|sigma_out|hamiltonian)\[([^\[\]]*)\]\.([xy])',lambda m:f'{m[1]}[2*({m[2]})+{1 if m[3]=="x"else 2}]',s)
 # Shift remaining Cflat/packed-vector subscripts from0-based to1-based.
 # Protect converted component accesses first.
 saved=[]
 def protect(m):saved.append(m[0]);return f'__MEM{len(saved)-1}__'
 s=re.sub(r'(?:sigma_in|sigma_out|hamiltonian)\[2\*\([^\[\]]*\)\+[12]\]',protect,s)
 s=re.sub(r'(sigma_in|sigma_out|hamiltonian)\[([^\[\]]*)\]',lambda m:f'{m[1]}[({m[2]})+1]',s)
 for j,v in enumerate(saved):s=s.replace(f'__MEM{j}__',v)
 s=s.replace('tmp.x','tmp_x').replace('tmp.y','tmp_y').replace('0.0','0f0').replace('hdt','HDT').replace('v(0f0)','StudyV4(0f0)').replace('v(hamiltonian','StudyV4(hamiltonian')
 s=s.replace('blockIdx.x','BX').replace('blockIdx.y','BY').replace('threadIdx.x','TX').replace('threadIdx.y','TY').replace('blockDim.x','BDX').replace('/','÷')
 return s

def statements(b):
 tokens=[];i=0;start=0;paren=0
 for i,ch in enumerate(b):
  if ch=='(':paren+=1
  elif ch==')':paren-=1
  elif ch in';{}'and paren==0:
   if b[start:i].strip():tokens.append(b[start:i].strip())
   if ch in'{}':tokens.append(ch)
   start=i+1
 if b[start:].strip():tokens.append(b[start:].strip())
 return tokens
rows=[];out=[]
for kid,(name,head,body)in enumerate(functions[:24]):
 lines=[];depth=1
 for t in statements(body):
  if t=='{':depth+=1;continue
  if t=='}':depth-=1;lines.append(' '*depth+'end');continue
  if t.startswith('for '):
   m=re.fullmatch(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*(.*?);\s*\1\s*<\s*(.*?);\s*(?:\+\+\1|\1\+\+)\s*\)',t);assert m,(name,t)
   lines.append(' '*depth+f'for {m[1]} in Int32({expr(m[2])}):Int32({expr(m[3])})-Int32(1)');continue
  if re.match(r'(?:const\s+)?real_(?:vec|2)?_?t\s*\*',t):continue
  if re.match(r'real_2_t\s+\w+$',t):continue # tmp.x/y declarations assignedbelow
  t=re.sub(r'^(?:const\s+)?(?:real_vec_t|real_t|int)\s+', '',t)
  if '='not in t:raise AssertionError((name,t))
  lines.append(' '*depth+expr(t))
 assert depth==1,(name,depth)
 code='@inline function '+name+'!(sigma_in,sigma_out,hamiltonian,dim::Int32,BX::Int32,BY::Int32,TX::Int32,TY::Int32,BDX::Int32)\n'+ ' @inbounds begin\n'+'\n'.join(lines)+'\n end\n return\nend\n'
 out.append(code);rows.append({'kid':kid,'native_name':name,'native_expanded_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'translation':'Mechanical brace/for/declaration/index conversion preserving activepreprocessor branch/native arithmetic/looporder; CUDAcoordinates passedbyownframeworkentry','runtime_dim':',\n    const int dim)'in head or 'const int dim)'in head,'manual_vector4':16<=kid<=23,'prefetch_source_comments_only':kid in[19,22]})
(p/'study_generated_bodies.jl').write_text('\n'.join(out));(p/'source_translation_map.json').write_text(json.dumps({'source':'src/hexciton-cuda/kernels.cu','preprocessor':'g++ -E -P -w -x c++ -DDIM=7 -DVEC_LENGTH=4 -DVEC_LENGTH_AUTO=4 -DPACKAGES_PER_WG=64 -DNUM_SUB_GROUPS=2 -DCHUNK_SIZE=16 -DWARP_SIZE=32','USE_INITZERO':'undefinedmatchingnative','entries':rows},indent=2)+'\n');print([(r['kid'],r['native_name'])for r in rows])
