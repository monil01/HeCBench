#!/usr/bin/env python3
import pathlib,re,json,hashlib
B=pathlib.Path(__file__).resolve().parent;R=B/'reference_history';D=B/'configured_balance';D.mkdir(exist_ok=True)
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
  name='"balance.'+base+'"' if label=='balance.cu' else '(sizeof(KeyT)==8?"hash64_32.'+base+'":"hash32_32.'+base+'")'
  out.extend([s[pos:m.start()],f'lr_launch({name},dim3({parts[0]}),dim3({parts[1]}),[&](){{ {original}; }}, {args})']);pos=j;rows.append(dict(source=label,line=s.count('\n',0,m.start())+1,kernel=kernel))
 out.append(s[pos:]);return '#include "../launch_bridge.h"\n'+''.join(out)

for name in ('balance.cu','hash_table.h'):
 (D/name).write_text(transform((R/name).read_text(),name))
(D/'balance.h').write_bytes((R/'balance.h').read_bytes())
(D/'provenance.json').write_text(json.dumps(dict(scope='OriginalcompletebalancePerformV2operator,hostlaunchbranchesonly',original_sha256={p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in ['balance.cu','hash_table.h','balance.h']},configured_sha256={p:hashlib.sha256((D/p).read_bytes()).hexdigest()for p in ['balance.cu','hash_table.h','balance.h']},launches=rows),indent=2)+'\n');print('Intercepted',len(rows),'launchsites')
