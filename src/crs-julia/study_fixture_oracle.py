from pathlib import Path
import struct,json,itertools,hashlib
p=Path(__file__).resolve().parent
POLYS={4:0x13,5:0x25,6:0x43,7:0x89,8:0x11d};MASK=(1<<64)-1

def mul(a,b,w):
 r=0
 while b:
  if b&1:r^=a
  b>>=1;a<<=1
  if a&(1<<w):a^=POLYS[w]
 return r

def power(a,n,w):
 r=1
 while n:
  if n&1:r=mul(r,a,w)
  a=mul(a,a,w);n>>=1
 return r

def invertible(a,w):
 a=[r[:]for r in a];n=len(a)
 for c in range(n):
  pivot=next((r for r in range(c,n)if a[r][c]),None)
  if pivot is None:return False
  a[c],a[pivot]=a[pivot],a[c];scale=power(a[c][c],(1<<w)-2,w);a[c]=[mul(x,scale,w)for x in a[c]]
  for r in range(n):
   if r!=c:
    factor=a[r][c];a[r]=[x^mul(factor,y,w)for x,y in zip(a[r],a[c])]
 return True

rows=[list(map(int,line.split()))for line in(p/'native_matrix_columns.txt').read_text().splitlines()];fixture=[];oracle=[];reviews=[]
for k,m,w,*bm in rows:
 coeff=[[power(r^(m+d),(1<<w)-2,w)for d in range(k)]for r in range(m)];expected=[]
 for d in range(k):
  for j in range(w):
   bits=0
   for r in range(m):bits|=mul(coeff[r][d],1<<j,w)<<(r*w)
   expected.append(bits)
 assert expected==bm,(k,m,w,'NativeGFMatrixDisagreement')
 generator=[[int(i==j)for j in range(k)]for i in range(k)]+coeff;subsets=list(itertools.combinations(range(k+m),k));assert all(invertible([generator[i]for i in subset],w)for subset in subsets)
 reviews.append({'k':k,'m':m,'w':w,'native_packed_columns_match_independent_polynomial_GF':True,'all_k_row_subsets_invertible':True,'tested_subsets':len(subsets)})
 for mode in [0,1]:
  size=w*(3 if mode==0 else 33);tasks=1 if mode==0 else 3;data=[]
  for task in range(tasks):
   for i in range(k*size):
    x=((i+1)*0x9e3779b97f4a7c15+(task+1)*0xd1b54a32d192ed03)&MASK;x^=x>>29;x=(x*0x94d049bb133111eb)&MASK
    data.append((0 if task==0 else MASK if task==1 else x)if mode==1 else x)
  fixture +=[k,m,w,size,tasks,mode]+bm+data
  for task in range(tasks):
   output=[0]*(m*size)
   for group in range(size//w):
    # Independent polynomial field product for each basis element of each64-bit plane.
    for d in range(k):
     for j in range(w):
      word=data[task*k*size+d*size+group*w+j]
      for r in range(m):
       product=mul(coeff[r][d],1<<j,w)
       for outbit in range(w):
        if product&(1<<outbit):output[r*size+group*w+outbit]^=word
   oracle +=output
(p/'fixtures_u64.bin').write_bytes(struct.pack('<'+'Q'*len(fixture),*fixture));(p/'oracle_u64.bin').write_bytes(struct.pack('<'+'Q'*len(oracle),*oracle))
(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent carryless polynomial multiply/reduce GF(2^w), exponentiation inverse for Cauchy coefficient1/(row XOR(m+column)); verify native bitmatrix columns then independently XOR polynomial products of64bitplanes','primitive_polynomials':POLYS,'matrix_reviews':reviews,'all50_mkw_matrix_contracts_pass':True,'all_MDS_k_row_subsets_tested':sum(r['tested_subsets']for r in reviews),'limits':['GPU only encodes, noGPUdecode orerrorcorrection pipeline claim','Host matrix generation is frozen source-generated fixture, not translated hostGF library','Nofloatingarithmetic/tolerance orstatisticalmodel']},indent=2)+'\n');print({'input_words':len(fixture),'output_words':len(oracle),'artifact_words':len(fixture)+len(oracle)+400})
