from pathlib import Path
import struct,json,hashlib
p=Path(__file__).resolve().parent;R=p.parent.parent

def oracle(q,t,mat,params):
 qlen,tlen,m,od,ed,oi,ei,w,bonus,zdrop,h0=params;profile=[mat[a*m+b]for a in range(m)for b in q];oldH=[0]*(qlen+1);oldE=[0]*(qlen+1);oldH[0]=h0;oldH[1]=max(h0-oi-ei,0)
 for j in range(2,qlen+1):
  if oldH[j-1]<=ei:break
  oldH[j]=oldH[j-1]-ei
 w=min(w,max((qlen*max(0,*mat)+bonus-oi)//ei+1,1),max((qlen*max(0,*mat)+bonus-od)//ed+1,1));best=h0;bi=bj=be=-1;gs=-1;mo=0;beg=0;end=qlen;trace=[]
 for i,a in enumerate(t):
  beg=max(beg,i-w);end=min(end,i+w+1,qlen);H=oldH.copy();E=oldE.copy();left=max(h0-od-ed*(i+1),0)if beg==0 else 0;f=0;rowbest=0;rowj=-1
  for j in range(beg,end):
   diag=oldH[j]+mat[a*m+q[j]]if oldH[j]else 0;cell=max(diag,oldE[j],f);H[j]=left;left=cell;E[j]=max(diag-od-ed,oldE[j]-ed,0);f=max(diag-oi-ei,f-ei,0)
   if cell>=rowbest:rowbest=cell;rowj=j
  H[end]=left;E[end]=0
  if end==qlen and left>=gs:gs=left;be=i
  oldH,oldE=H,E
  trace.append({'row':i,'beg':beg,'end':end,'row_max':rowbest})
  if rowbest==0:break
  if rowbest>best:best=rowbest;bi=i;bj=rowj;mo=max(mo,abs(rowj-i))
  elif zdrop>0 and best-rowbest-(max(i-bi-(rowj-bj),0)*ed+max(rowj-bj-(i-bi),0)*ei)>zdrop:break
  nz=[j for j in range(beg,end)if H[j]!=0 or E[j]!=0];beg=nz[0]if nz else end
  nz2=[j for j in range(beg,end+1)if H[j]!=0 or E[j]!=0];last=nz2[-1]if nz2 else beg-1;end=min(last+2,qlen)
 return[bi,bj],[bj+1,bi+1,be+1,gs,mo,best],oldH,oldE,profile,trace
fixtures=[];provenance=[]
for path in sorted((R/'src/extend2-sycl/testdata').glob('*.bin')):
 raw=path.read_bytes();i=0
 def integer():
  global i
  v=struct.unpack_from('<i',raw,i)[0];i+=4;return v
 nq=integer();q=list(raw[i:i+nq]);i+=nq;nt=integer();t=list(raw[i:i+nt]);i+=nt;m=integer();mat=list(struct.unpack_from('<25b',raw,i));i+=25;param=[nq,nt,m]+[integer()for _ in range(8)];expected=[integer()for _ in range(6)];_,got,H,E,qp,trace=oracle(q,t,mat,param);assert got==expected,(path.name,got,expected);fixtures.append((q,t,mat,param,got,H,E,qp));provenance.append({'path':str(path.relative_to(R)),'sha256':hashlib.sha256(raw).hexdigest(),'all6_stored_expected_match_independent_DP':True,'name':path.name,'params':param,'oracle':got,'rows_processed':len(trace)})
for name,q,t,w,z in [('match',[0,1,2,3]*3,[0,1,2,3]*3,99,0),('mismatch',[0]*7,[1]*11,99,0),('gap',[0,1,2,3]*3,[0,1,0,0,0,2,3]*3,99,0),('band',[0,1,2,3]*4,[3,2,1,0]*5,2,0),('zdrop',[0]*8,[0]*4+[1]*16,99,2)]:
 mat=[2 if a==b else-3 for a in range(5)for b in range(5)];param=[len(q),len(t),5,3,1,5,2,w,0,z,7];_,got,H,E,qp,trace=oracle(q,t,mat,param);fixtures.append((q,t,mat,param,got,H,E,qp));provenance.append({'name':name,'params':param,'oracle':got,'rows_processed':len(trace),'generated_fixture':True})
f=[];o=[]
for q,t,mat,param,got,H,E,qp in fixtures:f+=param+q+t+mat;o+=got+H+E+qp
(p/'fixtures_i32.bin').write_bytes(struct.pack('<%si'%len(f),*f));(p/'oracle_i32.bin').write_bytes(struct.pack('<%si'%len(o),*o));(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent Python row-snapshot affine DP recurrence (separate H/E row histories rather than in-place eh update), source band/zdrop and tie policy. All17external stored6-output golden records agree, plus5synthetic varied contracts. Scratch H/E and queryprofile fullreference derived. Intunboundedchecker excludesInt32overflow in namedfixtures.','cases':provenance,'case_count':len(fixtures),'input_prefix_count':len(f),'oracle_values':len(o)},indent=2)+'\n')
