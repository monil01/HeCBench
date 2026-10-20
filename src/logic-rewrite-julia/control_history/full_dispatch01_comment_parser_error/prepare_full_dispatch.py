#!/usr/bin/env python3
"""Full source host orchestration with every explicit kernel launch intercepted.
Original source archive unchanged. Non-null callbacks forbid native fallbacks.
Source Thrust host library calls remain native library interoperation, including
its predicates; this is a qualified hybrid and not a pure-framework claim.
"""
import pathlib,re,json,hashlib,shutil
B=pathlib.Path(__file__).resolve().parent;R=B/'reference_history';D=B/'configured_full';D.mkdir(exist_ok=True);rows=[]
pattern=re.compile(r'([A-Za-z_][\w:]*(?:\s*<[\w:,\s]+>)?)\s*<<<(.*?)>>>\s*\(',re.S)
for p in R.rglob('*'):
 if not p.is_file()or p.suffix in ('.o',)or p.name=='main':continue
 relative=p.relative_to(R);dest=D/relative;dest.parent.mkdir(parents=True,exist_ok=True);s=p.read_text()if p.suffix in ('.cu','.h','.cuh')else None
 if s is None:shutil.copy2(p,dest);continue
 pieces=[];pos=0
 for m in pattern.finditer(s):
  if m.start()<pos:continue
  depth=1;j=m.end();start=j
  while depth:depth+=(s[j]=='(')-(s[j]==')');j+=1
  args=s[start:j-1];cfg=m.group(2);parts=[];level=0;last=0
  for k,c in enumerate(cfg):
   level+=(c=='(')-(c==')')
   if c==','and level==0:parts.append(cfg[last:k].strip());last=k+1
  parts.append(cfg[last:].strip());assert len(parts)==2,(p,m.group(1),parts)
  kernel=m.group(1);base=kernel.split('<')[0].strip();original=s[m.start():j]
  label='hash64_32.' if p.name=='hash_table.h'else p.stem+'.'
  name='"'+label+base+'"'
  if p.name=='hash_table.h':name='(sizeof(KeyT)==8?"hash64_32.'+base+'":"hash32_32.'+base+'")'
  pieces.extend([s[pos:m.start()],f'lr_launch({name},dim3({parts[0]}),dim3({parts[1]}),[&](){{ {original}; }}, {args})']);pos=j
  rows.append(dict(source=str(relative),line=s.count('\n',0,m.start())+1,kernel=kernel,callback_name=name,original_launch_sha256=hashlib.sha256(original.encode()).hexdigest()))
 pieces.append(s[pos:]);out=''.join(pieces)
 if pos:out='#include "'+('../'*(len(relative.parts)))+'launch_bridge.h"\n'+out
 if p.name=='rewrite.cu':
  out+='''\n// Same-TU accessor avoids duplicate non-RDC managed registrations.
extern "C" void*lr_symbol(int id) noexcept {
 void*p=id==0?static_cast<void*>(&P):static_cast<void*>(&N);cudaPointerAttributes a{};
 cudaError_t e=cudaPointerGetAttributes(&a,p);
 if(e!=cudaSuccess||a.type!=cudaMemoryTypeManaged)return nullptr;
 return a.devicePointer;
}\n'''
 dest.write_text(out)
(D/'launch_provenance.json').write_text(json.dumps(dict(scope='Everyexplicituserlaunchcallback; nativeThrusthostlibraryinteropretainedanddisclosed; unselectedcallbacksrejectratherthanfallback',original_source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in R.rglob('*')if p.is_file()and p.suffix in('.cu','.h','.cuh')},configured_source_sha256={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest()for p in D.rglob('*')if p.is_file()and p.suffix in('.cu','.h','.cuh')},launches=rows),indent=2)+'\n');print('Intercepted',len(rows),'explicitlaunchsites')
