#!/usr/bin/env python3
"""Exact source parser-compatible CSR fixtures; no GPU execution."""
from pathlib import Path
import json,struct,collections,hashlib
P=Path(__file__).resolve().parent
cases=[]
for name,n,edges in [('path33',33,[(i,i+1)for i in range(32)]),('grid35',35,[(r*7+c,r*7+c+1)for r in range(5)for c in range(6)]+[(r*7+c,(r+1)*7+c)for r in range(4)for c in range(7)]),('star1057',1057,[(0,i)for i in range(1,1057)])]:
 labels=[f'n{i:05d}'for i in range(n)];f=P/(name+'.txt');f.write_text(''.join(f'{labels[u]} {labels[v]}\n'for u,v in edges));cases.append((name,f))
cases.append(('breast_coexpress',P/'dataset_readiness/breast_coexpress.txt'))
flat=[len(cases)];meta=[]
for name,path in cases:
 rows=[line.split()for line in path.read_text().splitlines()if line and line[0]not in '%#'];labels=sorted({v for row in rows for v in row});mapping={v:i for i,v in enumerate(labels)};edges=[(mapping[u],mapping[v])for u,v in rows];n=len(labels);m=len(edges);assert all(u!=v for u,v in edges);assert len({tuple(sorted(e))for e in edges})==m;directed=sorted(edges+[(v,u)for u,v in edges]);F=[u for u,v in directed];C=[v for u,v in directed];R=[0];k=0
 for u in range(n):
  while k<len(F)and F[k]==u:k+=1
  R.append(k)
 assert R[-1]==2*m;flat += [n,m]+R+C+F
 degree=[R[i+1]-R[i]for i in range(n)];meta.append(dict(name=name,n=n,m=m,path=str(path.relative_to(P)),source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),degree_max=max(degree),degree_mean=sum(degree)/n,IDs=labels))
(P/'fixtures.bin').write_bytes(struct.pack('<'+'i'*len(flat),*flat));(P/'input_manifest.json').write_text(json.dumps({'cases':meta,'fixture_bytes':4*len(flat),'fixture_sha256':hashlib.sha256((P/'fixtures.bin').read_bytes()).hexdigest()},indent=2)+'\n');print([{k:c[k]for k in ['name','n','m','degree_max']}for c in meta]);print('fixture_bytes',4*len(flat))
