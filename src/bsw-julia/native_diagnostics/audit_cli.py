#!/usr/bin/env python3
"""Independent full-matrix affine-gap Smith-Waterman and endpoint tie control."""
import pathlib,json,re,sys,hashlib
R=pathlib.Path(__file__).resolve().parents[3]
s=(R/'src/bsw-cuda/main.cu').read_text();mat=list(map(int,re.search(r'short scores_matrix\[\]\s*=\s*\{([^}]+)\}',s,re.S)[1].replace('\n','').split(',')));assert len(mat)==576
s=(R/'src/bsw-cuda/driver.cu').read_text();enc=list(map(int,re.search(r'short h_encoding_matrix\[\]\s*=\s*\{([^}]+)\}',s,re.S)[1].replace('\n','').split(',')));assert len(enc)==91

def sw(a,b,inverse=False):
 # row=longer, column=shorter, matching declared endpoint orientation, not GPU antidiagonal traversal.
 long,short=(a,b)if len(a)>=len(b)else(b,a)
 h=[[0]*(len(short)+1)for _ in range(len(long)+1)];e=[[float('-inf')]*(len(short)+1)for _ in h];f=[[float('-inf')]*(len(short)+1)for _ in h];best=0;locations=[]
 for i,c in enumerate(long,1):
  for j,d in enumerate(short,1):
   e[i][j]=max(e[i][j-1]-1,h[i][j-1]-6);f[i][j]=max(f[i-1][j]-1,h[i-1][j]-6)
   h[i][j]=max(0,e[i][j],f[i][j],h[i-1][j-1]+mat[enc[ord(c)]*24+enc[ord(d)]])
   if h[i][j]>best:best=h[i][j];locations=[(i,j)]
   elif h[i][j]==best:locations.append((i,j))
 assert best>0
 i,j=min(locations,key=(lambda ij:(ij[1],ij[0]))if inverse else(lambda ij:ij))
 return int(best),(i,j)if len(a)>=len(b)else(j,i),len(locations)

if __name__=='__main__':
 p=pathlib.Path(sys.argv[1]);fix=json.loads((p/'fixtures/fixture.json').read_text());actual=[list(map(int,x.split()))for x in (p/'original_results.tsv').read_text().splitlines()];assert len(actual)==len(fix['cases']);rows=[]
 for k,((a,b),gpu)in enumerate(zip(fix['cases'],actual)):
  score,(ae,be),ties=sw(a,b);back,(ac,bc),bties=sw(a[:ae][::-1],b[:be][::-1],True);assert score==back
  expected=[score,ae-ac,ae-1,be-bc,be-1];rows.append(dict(pair=k,lengths=[len(a),len(b)],expected=expected,actual=gpu,pass_exact=expected==gpu,forward_maximum_cells=ties,reverse_maximum_cells=bties))
 out=dict(benchmark='bsw',scope='Named65positiveaminoacidpairs; standard row-major full-matrix affine-gap DP, independently inferred source endpoint tie policy.',pass_exact=all(x['pass_exact']for x in rows),pairs=len(rows),score_pass_count=sum(x['actual'][0]==x['expected'][0]for x in rows),allfive_fields_pass_count=sum(x['pass_exact']for x in rows),original_output_sha256=hashlib.sha256((p/'original_results.tsv').read_bytes()).hexdigest(),cases=rows)
 (p/'original_cpu_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='cases'}));sys.exit(0 if out['pass_exact']else 1)
