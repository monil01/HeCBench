#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys,argparse,json
P=Path(__file__).resolve().parent;R=P.parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);a=parser.parse_args()
for name in ['fetch_original_assets.py','generate_assets.py','build_contract.py']:subprocess.run([sys.executable,str(P/name)],check=True)
d=json.loads((P/'contract.json').read_text());j=''
for name,key in [('WS','width'),('HS','height'),('KWS','kernel_width'),('KHS','kernel_height'),('NS','sad_count')]:j+='const '+name+'=Int32'+str([c[key] for c in d['cases']])+'\n'
j+=f'const COUNT={d["count"]};const PREFIX={d["prefix"]}\n';(P/'study_params.jl').write_text(j)
c=json.loads((P/'study_config.json').read_text())
def relocate(x):
 if isinstance(x,str):return x.replace('/home/imo/HeCBench_julia',str(R))
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items() if not k.startswith('reuse_')}
 return x
a.output.write_text(json.dumps(relocate(c),indent=2)+'\n');print(a.output)
