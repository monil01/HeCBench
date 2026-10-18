from pathlib import Path
import json,struct
p=Path(__file__).resolve().parent
cases=json.loads((p/'fixture_cases.json').read_text());words=[len(cases)]
for c in cases:
 h=len(c['t']);v=len(c['q']);d=h-v;c['seed']=[h,v,h+17,v+17,17,d,d,d,d,0]
 words += [v,h,c['drop']]+c['seed']+list(c['q'].encode())+list(c['t'].encode())
(p/'fixtures_i32.bin').write_bytes(struct.pack('<'+'i'*len(words),*words))
(p/'fixture_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(p/'oracle_i32.bin').write_bytes(struct.pack('<'+'i'*(5*len(cases)),*[v for c in cases for v in c['expected']]))
