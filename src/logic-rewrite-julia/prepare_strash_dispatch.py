#!/usr/bin/env python3
"""Source-preserving host launch interception for the full strash operator.
CUDAjl callbacks cannot fall back to native kernels. Native kernels remain only
for the reference branch. Original AIGMan wrapper is excluded from this unit.
"""
import pathlib,re,json,hashlib
B=pathlib.Path(__file__).resolve().parent;R=B/'reference_history';D=B/'configured_strash';D.mkdir(exist_ok=True)
rows=[]
def transform(s,label):
 pattern=re.compile(r'([A-Za-z_][\w:]*(?:\s*<[\w:,\s]+>)?)\s*<<<(.*?)>>>\s*\(',re.S);out=[];pos=0
 for m in pattern.finditer(s):
  if m.start()<pos:continue
  depth=1;j=m.end();start=j
  while depth:depth+=(s[j]=='(')-(s[j]==')');j+=1
  args=s[start:j-1];cfg=m.group(2);parts=[];level=0;last=0
  for k,c in enumerate(cfg):
   level+=(c=='(')-(c==')')
   if c==',' and level==0:parts.append(cfg[last:k].strip());last=k+1
  parts.append(cfg[last:].strip());assert len(parts)==2
  kernel=m.group(1);base=kernel.split('<')[0].strip();original=s[m.start():j]
  name='"strash.'+base+'"' if label=='strash.cu' else '(sizeof(KeyT)==8?"hash64_32.'+base+'":"hash32_32.'+base+'")'
  out.extend([s[pos:m.start()],f'lr_launch({name},dim3({parts[0]}),dim3({parts[1]}),[&](){{ {original}; }}, {args})']);pos=j;rows.append(dict(source=label,line=s.count('\n',0,m.start())+1,kernel=kernel))
 out.append(s[pos:]);return '#include "../launch_bridge.h"\n'+''.join(out)
s=(R/'strash.cu').read_text();s=s[:s.index('void AIGMan::strash')];(D/'strash.cu').write_text(transform(s,'strash.cu'))
(D/'hash_table.h').write_text(transform((R/'hash_table.h').read_text(),'hash_table.h'))
(D/'strash.cuh').write_bytes((R/'strash.cuh').read_bytes())
(D/'provenance.json').write_text(json.dumps(dict(scope='CompleteAig::strashhostpipeline+allitsauthoredkernelcallbackbranches, originalAIGManwrappernotinthisunit',original_sha256={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in ['strash.cu','hash_table.h','strash.cuh']},configured_sha256={p:hashlib.sha256((D/p).read_bytes()).hexdigest()for p in ['strash.cu','hash_table.h','strash.cuh']},launches=rows),indent=2)+'\n');print('Intercepted',len(rows),'launchsites')
