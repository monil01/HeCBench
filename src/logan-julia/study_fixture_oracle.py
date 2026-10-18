from pathlib import Path
import json,hashlib,struct,random,shutil
r=Path(__file__).resolve().parents[2];p=r/'src/logan-julia';a=p/'logan-julia-cuda.jl';a.mkdir(exist_ok=True)
for f in ['main.jl','Makefile']:
 if (p/f).exists() and not (a/('original_'+f)).exists():shutil.copy2(p/f,a/('original_'+f))
shutil.copy2(r/'src/logan-cuda/LICENSE',p/'LICENSE')
# All source calls have both nonempty directions; lengths across thread/warp boundaries.
rng=random.Random(731);cases=[];allwords=[];oracle=[]
for ci,(n,drop) in enumerate([(7,100),(33,100),(65,100),(129,100),(47,100)]):
 q=[rng.choice('ACGT') for _ in range(n)];t=q.copy()
 for j in range(ci,n,13):t[j]=rng.choice([x for x in 'ACGT' if x!=q[j]])
 # equal segment lengths, no-pruning guaranteed because global path >= -n, actual best never below 0;
 # n129 has at most10 mismatches and all prefixes have >=-1 through a diagonal match witness.
 cases.append(dict(n=n,drop=drop,q=''.join(q),t=''.join(t),seed=[11,7,28,24,17,4,4,4,4,0]))
# Explicit tiny all-mismatch and unequal lengths exercise gap boundaries.
cases+= [dict(n=5,drop=100,q='AAAAA',t='CCCCC',seed=[9,5,26,22,17,4,4,4,4,0]),dict(n=19,drop=100,q='ACGT'*4+'ACG',t='ACGT'*3+'AC',seed=[13,3,30,20,17,10,10,10,10,0])]
# Independent dense Needleman-Wunsch global recurrence. With drop100 and these controlled diagonals,
# no cell on selected optimal end path is pruned. Track source active-bound diagonal extrema separately.
# Oracle endpoint score is full (lenq,lent) score; endpoint/diag source-policy checked by separate recurrence below.
def source_policy(q,t,drop):
 rows=len(t)+1;cols=len(q)+1;U=-32767;a1=[];a2=[0];a3=[-1,-1];o1=o2=o3=0;d=1;lo=0;hi=0;best=0;mn=1;mx=2
 while mn<mx:
  d+=1;a1,a2,a3=a2,a3,a1;o1,o2,o3=o2,o3,mn-1;a3=[U]*(mx+1-o3)
  if -d>best-drop:
   if o3==0:a3[0]=-d
   if d-mx==0:a3[mx-o3]=-d
  for c in range(mn,mx):
   v=max(a2[c-o2],a2[c-o2-1])-1;v=max(v,a1[c-o1-1]+(1 if q[c-1]==t[c+rows-d-1] else -1));a3[c-mn+1]=U if v<best-drop else v
  best=max(best,max(a3))
  while mn-o3<len(a3) and a3[mn-o3]==U and mn-o2-1<len(a2) and a2[mn-o2-1]==U:mn+=1
  while mx-o3>0 and a3[mx-o3-1]==U and a2[mx-o2-1]==U:mx-=1
  mx+=1;lo=min(lo,mn-(d-mn));hi=max(hi,mx-2-(d+2-mx));mn=max(mn,d+2-rows);mx=min(mx,cols)
 c=len(a3)+o3-2;rr=d-c;s=a3[c-o3]
 if s==U:
  if a2[-2]!=U:c=len(a2)+o2-2;rr=d-1-c;s=a2[-2]
  elif len(a2)>2 and a2[-3]!=U:c=len(a2)+o2-3;rr=d-1-c;s=a2[-3]
 if s==U:
  for i,v in enumerate(a1):
   if v>s:s=v;c=i+o1;rr=d-2-c
 return s,c,rr,lo,hi
for c in cases:
 # Native query/database mapping t indexes backwards by antidiagonal. Full DP target must reverse stored t.
 q=c['q'];t=c['t'][::-1];D=[[0]*(len(t)+1) for _ in range(len(q)+1)]
 for i in range(len(q)+1):D[i][0]=-i
 for j in range(len(t)+1):D[0][j]=-j
 for i in range(1,len(q)+1):
  for j in range(1,len(t)+1):D[i][j]=max(D[i-1][j]-1,D[i][j-1]-1,D[i-1][j-1]+(1 if q[i-1]==t[j-1] else -1))
 s,col,row,lo,hi=source_policy(c['q'],c['t'],c['drop'])
 assert (col,row)==(len(q),len(t)),(c,s,col,row)
 assert s==D[-1][-1],(s,D[-1][-1])
 c['expected']=[s,col,row,lo,hi]
allwords=[len(cases)]
for c in cases:allwords += [len(c['q']),len(c['t']),c['drop']]+c['seed']+list(c['q'].encode())+list(c['t'].encode())
(p/'fixtures_i32.bin').write_bytes(struct.pack('<'+'i'*len(allwords),*allwords))
(p/'fixture_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(p/'independent_oracle.json').write_text(json.dumps(dict(method='Independent dense Needleman-Wunsch integer global DP on query and reversed stored target; full endpoint scores crosschecked source-policy anti-diagonal endpoint/diagonal tracking independently on CPU',cases=cases,scope='Named nonpruned finite segments only; no adaptive-pruning/long-read quality certificate'),indent=2)+'\n')
