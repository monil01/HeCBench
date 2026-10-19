#!/usr/bin/env python3
from pathlib import Path
import re,array,hashlib,json,sys,tarfile
P=Path(__file__).resolve().parent;R=P.parent.parent;D=R/'.porting-state/diagnostics/grep/external_dataset';original=(D/'testcases/romeojuliet.txt').read_bytes();assert original.endswith(b'\n')and b'\0'not in original
patterns=[b'ROMEO|JULIET',b'R+R*',b'(R|J)ULIET',b'R..EO',b'[a-b]',b'[0-9]',rb'\.\?\|\+\*',b'(ab)?c+',b'.*',rb't*hi.\?']
# Sourcehost token simplification, checked independently against originalhost simplifyRe.
OPS={ord('+'):1,ord('?'):2,ord('*'):3,ord('|'):4,ord('('):5,ord(')'):6,ord('.'):21}
def simplify(x):
 out=[];i=0
 while i<len(x):
  c=x[i]
  if c==92:i+=1;out.append(x[i])
  elif c==91:
   assert x[i+2]==45 and x[i+4]==93;a,b=x[i+1],x[i+3];out+=[5,a]
   for v in range(a+1,b+1):out+=[4,v]
   out+=[6];i+=4
  else:out.append(OPS.get(c,c))
  i+=1
 return bytes(out+[0])
strings=[b'ROMEO',b'JULIET',b'R',b'RRRR',b'RJULIET',b'RaXEO',b'a',b'b',b'5',b'.?|+*',b'abccc',b'ccc',b'hiX?',b'ttthiZ?',b'',b'ROMEOJULIET',b'other']
corpora=[[b'ROMEO'],original[:-1].split(b'\n'),[strings[i%len(strings)]for i in range(81923)]]
prefix=[3,len(patterns),512,160,100,1000,8000,256];tail=[];cases=[];files={}
for j,pat in enumerate(patterns):
 simp=simplify(pat);(P/f'regex{j}_u8.bin').write_bytes(simp);prefix+=[j,len(simp)]+[211]*16+list(simp)+[211]*16
for k,lines in enumerate(corpora):
 text=bytearray();offsets=[]
 for line in lines:assert b'\0'not in line;offsets.append(len(text));text+=line+b'\0'
 (P/f'text{k}_u8.bin').write_bytes(text);(P/f'offset{k}_u32.bin').write_bytes(array.array('I',offsets).tobytes())
 prefix+=[k,len(lines),len(text)]+[211]*16+list(text)+[211]*16+[1234567]*16+offsets+[1234567]*16
 matches=[]
 for j,pat in enumerate(patterns):
  values=[int(re.fullmatch(pat,line)is not None)for line in lines];matches.append(sum(values));prefix+=[211]*16+[0]*len(lines)+[211]*16;tail+=[211]*16+values+[211]*16
 cases.append({'case':k,'lines':len(lines),'text_bytes':len(text),'matched_counts_by_pattern':matches,'source':'actualRomeoJulietfullcorpus'if k==1 else'namedboundary/gridstridecontrol'})
(P/'oracle_i32.bin').write_bytes(array.array('i',prefix+tail).tobytes())
for f in P.glob('*.bin'):files[f.name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
plan={'benchmark':'grep','patterns':[x.decode()for x in patterns],'corpora':cases,'source_blocks':512,'source_threads':160,'prefix_count':len(prefix),'artifact_count':len(prefix+tail),'atol':0,'rtol':0,'oracle':'IndependentPythonre.fullmatchbyteengine, not ThompsonGPUgraphconstruction','ROI':'30resident originalGPU regexparse/postfix/NFAcompile+full-stringmatching launches andwholedevicecompletion','full_original_host_CLI_frontend_translated':False,'native_reference_repaired':False,'native_integration_adapted':True,'exclude_untouched_native_aggregates':True,'source_domain_bounds':'ValidASCIIregex/tokenstream,states<=100,Frags<=1000,postfix<8000,list/epsilonstack<=100; no nullableepsiloncycles, explicitcapacityauditrequired','files':files,'python':sys.version,'external_corpus_sha256':hashlib.sha256(original).hexdigest()}
(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');(P/'study_params.h').write_text(f'#define PREFIX {len(prefix)}\n#define COUNT {len(prefix+tail)}\n#define PATTERNS 10\n#define CASES 3\n');(P/'study_params.jl').write_text(f'const PREFIX={len(prefix)}\nconst COUNT={len(prefix+tail)}\nconst PATTERNS=10\nconst CASES=3\n');print(json.dumps({'cases':cases,'prefix':len(prefix),'count':len(prefix+tail)},indent=2))
