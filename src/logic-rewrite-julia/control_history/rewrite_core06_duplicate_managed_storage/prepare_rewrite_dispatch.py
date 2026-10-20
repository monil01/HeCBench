#!/usr/bin/env python3
"""Replace only host kernel launch sites with audited callback/native branches.
Non-null callbacks never execute the original native kernel; errors propagate.
Original source files remain unchanged. Includes/library data remain source-bound.
"""
import argparse,pathlib,re,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;src=B/'reference_history/rewrite.cu';s=src.read_text();pattern=re.compile(r'([A-Za-z_][\w:]*)\s*<<<(.*?)>>>\s*\(',re.S);pieces=[];pos=0;sites=[]
for m in pattern.finditer(s):
 if m.start()<pos:continue
 depth=1;j=m.end();start=j
 while depth:
  c=s[j];depth+=(c=='(')-(c==')');j+=1
 args=s[start:j-1];config=m.group(2);configs=[];level=0;previous=0
 for k,c in enumerate(config):
  level+=(c=='(')-(c==')')
  if c==','and level==0:configs.append(config[previous:k].strip());previous=k+1
 configs.append(config[previous:].strip());assert len(configs)==2
 kernel=m.group(1);original=s[m.start():j]
 replacement=f'lr_launch("{kernel}",dim3({configs[0]}),dim3({configs[1]}),[&](){{ {original}; }}, {args})'
 pieces.append(s[pos:m.start()]);pieces.append(replacement);pos=j
 sites.append(dict(kernel=kernel,source_line=s.count('\n',0,m.start())+1,configuration=configs,arguments=args))
pieces.append(s[pos:]);out='#include "../launch_bridge.h"\n'+''.join(pieces);a.output.mkdir(parents=True,exist_ok=True);f=a.output/'rewrite.cu';f.write_text(out)
(a.output/'rewrite_launch_provenance.json').write_text(json.dumps(dict(original=str(src.relative_to(B)),original_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),configured_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),scope='Host launch callback branch only; source mathematicaldevicebodies/CPUalgorithms unchanged. Non-null callback never fallsback. Nativebranch existsonlyforreference. UnselectedWavekernel callbacks rejected.',sites=sites),indent=2)+'\n');print('Rebound launchsites',len(sites))
