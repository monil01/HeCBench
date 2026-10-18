from pathlib import Path
import re,json,struct,math
p=Path(__file__).resolve().parent;F=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
s=(p/'original_native/matrix.h').read_text().split('#else')[0];H=[list(map(int,re.findall(r'-?\d+',row))) for row in re.findall(r'\{([^{}]+)\}',s) if len(re.findall(r'-?\d+',row))==24];assert len(H)==12
checks=[[c*96+(r+shift)%96 for c,shift in enumerate(row) if shift>=0] for row in H for r in range(96)];vedges=[[] for _ in range(2304)]
for j,row in enumerate(checks):
 for k,v in enumerate(row):vedges[v].append((j,k))
count1=[sum(s>=0 for s in row) for row in H];count2=[sum(H[r][c]>=0 for r in range(12)) for c in range(24)];pack=lambda r,c,s,v:(r&255)|((c&255)<<8)|((s&255)<<16)|((v&255)<<24)
comp1=[pack(0,0,-1,0)]*84;comp2=[pack(0,0,-1,0)]*288
for r,row in enumerate(H):
 for k,(c,s) in enumerate((c,s) for c,s in enumerate(row) if s>=0):comp1[k*12+r]=pack(r,c,s,1)
for c in range(24):
 for k,r in enumerate(r for r in range(12) if H[r][c]>=0):comp2[k*24+c]=pack(r,c,H[r][c],1)
def encode(seed):
 info=[int((i*7+seed*11)%29<13) for i in range(1152)];basis={}
 for row in checks:
  bits=0;rhs=0
  for v in row:
   if v<1152:rhs^=info[v]
   else:bits^=1<<(v-1152)
  while bits:
   pv=(bits&-bits).bit_length()-1
   if pv in basis:bb,rr=basis[pv];bits^=bb;rhs^=rr
   else:basis[pv]=(bits,rhs);break
  if not bits:assert rhs==0
 assert len(basis)==1152;sol=0
 for pv,(bits,rhs) in sorted(basis.items(),reverse=True):
  bit=rhs^((bits&sol).bit_count()&1)
  if bit:sol|=1<<pv
 code=info+[sol>>i&1 for i in range(1152)];assert all(sum(code[v] for v in row)%2==0 for row in checks);return code
prefix=[3]+[x for row in H for x in row]+count1+count2+comp1+comp2;gold=[];cases=[];guard=-12345.;profiles=[]
for ci,ncw in enumerate([2,4,2]):
 all_llr=[];codes=[];alls=[];stateR=[];stateDT=[];hard=[];syndromes=[];errors=[];maxapp=0.;minmargin=math.inf
 for cw in range(ncw):
  code=encode(7+cw) if ci<2 else [0]*2304;codes+=code
  llr=[F((1 if bit==0 else -1)*(1+(i%13)/16)) for i,bit in enumerate(code)]
  if ci==1:
   for i in [cw*3,17+cw*11,113+cw*7,307+cw*13]:llr[i]=F((1 if code[i] else -1)*.125)
  if ci==2:llr=[0.]*2304
  all_llr+=llr;app=llr[:];old=[[0.]*len(row) for row in checks];denseR=[0.]*27648;denseDT=[0.]*27648;hd=[]
  for it in range(10):
   delta=[[0.]*len(row) for row in checks]
   for j,row in enumerate(checks):
    q=[app[v] if it==0 else F(app[v]-old[j][k]) for k,v in enumerate(row)];new=[]
    # Independent leave-one-edge enumeration, not source two-min/tie-index code.
    for k in range(len(row)):
     others=[x for z,x in enumerate(q) if z!=k];sgn=-1. if sum(x<0 for x in others)%2 else 1.;val=F(F(.75*sgn)*min([1000.]+[abs(x) for x in others]));new.append(val);delta[j][k]=val if it==0 else F(val-old[j][k])
     rb=j//96;sub=j%96;col=row[k]//96;addr=col*1152+rb*96+sub;denseR[addr]=val;denseDT[addr]=delta[j][k]
    old[j]=new
   nextapp=[]
   for v,conns in enumerate(vedges):
    val=app[v]
    for j,k in conns:val=F(val+delta[j][k])
    nextapp.append(val)
   if it<9:app=nextapp
   else:hd=[0 if v>0 else 1 for v in nextapp];minmargin=min(minmargin,min(abs(v) for v in nextapp))
   maxapp=max(maxapp,max(map(abs,nextapp)))
  alls+=app;stateR+=denseR;stateDT+=denseDT;hard+=hd;syndromes.append(sum(sum(hd[v] for v in row)%2 for row in checks));errors.append(sum(x!=y for x,y in zip(hd,code)))
 prefix +=[ncw]+all_llr+codes
 for a in [alls,stateDT,stateR,hard]:gold+=[guard]*8+a+[guard]*8
 gold +=[guard]*8+all_llr+[guard]*8
 gold +=[90]*8+count1+[90]*8+[90]*8+count2+[90]*8
 gold +=[1515870810]*8+comp1+[1515870810]*8+[1515870810]*8+comp2+[1515870810]*8
 cases.append(dict(case=ci,codewords=ncw,macrocodewords=ncw//2,mode=['noiseless_nonzero_codewords','four_weak_sign_errors_per_codeword','neutral_zero_LLR_tie_control'][ci],output_sizes=[ncw*2304,ncw*27648,ncw*27648,ncw*2304],syndrome_weights=syndromes,decoded_bit_errors_vs_transmitted=errors,max_abs_APP=maxapp,min_final_abs_APP=minmargin,known_codeword_claim=ci<2))
for name,a in [('fixture.bin',prefix),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),graph=dict(mode='WIMAX802.16e',Z=96,block_rows=12,block_cols=24,check_nodes=1152,variable_nodes=2304,edges=sum(map(len,checks)),row_degrees=count1,column_block_degrees=count2,parity_GF2_rank=1152),oracle='Independent full Tanner graph leave-one-edge normalizedminsum with explicitFloat32rounding; GF2Gaussian parityencoding andsyndrome checks. Sourcefixed10iterations, LLRcapturedafter9normalupdates, lastVNPemitsHDwithoutLLRwrite.',atol=0,rtol=0,scope='Allfullgraphmessage/statearrays, inactiveR/dt slotszeroedonlyforcapture; named2/4codewordbatches, noAWGNBERSNRscientificclaim/500repeatCLI'),indent=2)+'\n')
print(json.dumps(cases))
