from pathlib import Path
import re,sys
sys.path.insert(0,'/tmp');import atomic_owned
src=Path('src/secp256k1-cuda/main.cu').read_text();p=Path('src/secp256k1-julia')
def body(name):
 start=src.index('void '+name+'(');start=src.index('{',start);depth=1;i=start+1
 while depth:
  if src[i]=='{':depth+=1
  if src[i]=='}':depth-=1
  i+=1
 return re.sub(r'/\*.*?\*/','',src[start+1:i-1],flags=re.S)
def expr(s):
 s=s.strip().replace("a->n[","a[")
 while '(ulong)' in s:
  a=s.index('(ulong)');j=a+7
  while s[j].isspace():j+=1
  if s[j]=='(':
   depth=1;k=j+1
   while depth:
    if s[k]=='(':depth+=1
    if s[k]==')':depth-=1
    k+=1
   value=s[j:k]
  else:
   m=re.match(r'[A-Za-z_][A-Za-z_0-9]*(?:\[\d+\])?',s[j:]);assert m;sym=m[0];k=j+len(sym);value=sym
  s=s[:a]+'UInt64('+value+')'+s[k:]
 placeholders={}
 def arr(m):
  key='ARR'+chr(65+len(placeholders));placeholders[key]=m[1]+'['+str(int(m[2])+1)+']';return key
 s=re.sub(r'([ab])\[(\d+)\]',arr,s)
 def hexx(m):
  key='HEX'+chr(65+len(placeholders));placeholders[key]=('UInt64' if m[2] else 'UInt32')+'('+m[1]+')';return key
 s=re.sub(r'(0x[0-9a-fA-F]+)(UL)?',hexx,s)
 s=re.sub(r'\b\d+\b',lambda m:'UInt32('+m[0]+')',s)
 for k,v in placeholders.items():s=s.replace(k,v)
 return s
out=['# Direct scalar transcription of original secp256k1 10x26 field arithmetic.\n# UInt32/UInt64 fixed-width wrap and unsigned casts preserve native semantics.\n']
for cname,jname,args,nret,rettype in [('secp256k1_fe_sqr_inner','fe_sqr','a::NTuple{10,UInt32}',10,'UInt32'),('secp256k1_fe_mul_inner','fe_mul','a::NTuple{10,UInt32},b::NTuple{10,UInt32}',10,'UInt32'),('secp256k1_fe_get_b32','fe_bytes','a::NTuple{10,UInt32}',32,'UInt8')]:
 types={};lines=[]
 for st in body(cname).split(';'):
  st=st.strip()
  if not st:continue
  decl=re.match(r'^(const\s+)?(uint|ulong)\s+(.+)$',st,re.S)
  if decl:
   typ={'uint':'UInt32','ulong':'UInt64'}[decl[2]]
   for term in decl[3].split(','):
    v=term.strip();name=v.split('=')[0].strip();types[name]=typ
    if '=' in v:lines.append(' '+name+' = ('+expr(v.split('=',1)[1])+') % '+typ)
   continue
  a=re.match(r'^(r\[\d+\]|\w+)\s*(>>=|<<=|\+=|-=|&=|\|=|=)\s*(.*)$',st,re.S)
  assert a,(cname,st)
  lhs,op,rhs=a.groups();idx=re.match(r'r\[(\d+)\]',lhs)
  if idx:lhs='r'+idx[1];typ=rettype
  else:typ=types[lhs]
  v=expr(rhs)
  if op!='=':v=lhs+' '+op[:-1]+' ('+v+')'
  lines.append(' '+lhs+' = ('+v+') % '+typ)
 out.append('function '+jname+'('+args+')\n'+'\n'.join(lines)+'\n ('+','.join('r'+str(i) for i in range(nret))+',)\nend\n')
out.append('''@inline function fe_storage(a::NTuple{8,UInt32})
 m=UInt32(0x3ffffff)
 (a[1]&m,(a[1]>>26)|((a[2]<<6)&m),(a[2]>>20)|((a[3]<<12)&m),(a[3]>>14)|((a[4]<<18)&m),(a[4]>>8)|((a[5]<<24)&m),(a[5]>>2)&m,(a[5]>>28)|((a[6]<<4)&m),(a[6]>>22)|((a[7]<<10)&m),(a[7]>>16)|((a[8]<<16)&m),a[8]>>10)
end
@inline function fe_weak(a::NTuple{10,UInt32})
 m=UInt32(0x3ffffff);x=a[10]>>22
 t0=a[1]+x*UInt32(0x3d1);t1=a[2]+(x<<6)+(t0>>26);t2=a[3]+(t1>>26);t3=a[4]+(t2>>26);t4=a[5]+(t3>>26);t5=a[6]+(t4>>26);t6=a[7]+(t5>>26);t7=a[8]+(t6>>26);t8=a[9]+(t7>>26);t9=(a[10]&UInt32(0x3fffff))+(t8>>26)
 (t0&m,t1&m,t2&m,t3&m,t4&m,t5&m,t6&m,t7&m,t8&m,t9)
end
@inline fe_add(a::NTuple{10,UInt32},b::NTuple{10,UInt32})=ntuple(i->a[i]+b[i],Val(10))
@inline fe_int(a::NTuple{10,UInt32},n::UInt32)=ntuple(i->a[i]*n,Val(10))
@inline function fe_neg(a::NTuple{10,UInt32},m::UInt32)
 ntuple(i->(i==1 ? UInt32(0x3fffc2f) : i==2 ? UInt32(0x3ffffbf) : i==10 ? UInt32(0x3fffff) : UInt32(0x3ffffff))*UInt32(2)*(m+UInt32(1))-a[i],Val(10))
end
@inline fe_one()=(UInt32(1),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0))
function fe_squares(a,n)
 for _ in 1:n;a=fe_sqr(a);end
 a
end
function fe_inv(a)
 x2=fe_mul(fe_sqr(a),a);x3=fe_mul(fe_sqr(x2),a)
 x6=fe_mul(fe_squares(x3,3),x3);x9=fe_mul(fe_squares(x6,3),x3);x11=fe_mul(fe_squares(x9,2),x2)
 x22=fe_mul(fe_squares(x11,11),x11);x44=fe_mul(fe_squares(x22,22),x22);x88=fe_mul(fe_squares(x44,44),x44)
 x176=fe_mul(fe_squares(x88,88),x88);x220=fe_mul(fe_squares(x176,44),x44);x223=fe_mul(fe_squares(x220,3),x3)
 t=fe_mul(fe_squares(x223,23),x22);t=fe_mul(fe_squares(t,5),a);t=fe_mul(fe_squares(t,3),x2);fe_mul(a,fe_squares(t,2))
end
function ge_add(x,y,z,bx,by)
 z12=fe_sqr(z);u1=fe_weak(x);u2=fe_mul(bx,z12);s1=fe_weak(y);s2=fe_mul(fe_mul(by,z12),z)
 h=fe_add(fe_neg(u1,UInt32(1)),u2);ii=fe_add(fe_neg(s1,UInt32(1)),s2)
 i2=fe_sqr(ii);h2=fe_sqr(h);h3=fe_mul(h,h2);znew=fe_mul(z,h);t=fe_mul(u1,h2)
 xx=fe_add(fe_neg(fe_add(fe_int(t,UInt32(2)),h3),UInt32(3)),i2)
 yy=fe_mul(fe_add(fe_neg(xx,UInt32(5)),t),ii);yy=fe_add(yy,fe_neg(fe_mul(h3,s1),UInt32(1)))
 xx,yy,znew
end
@inline function load_point(input,offset)
 @inbounds x=fe_storage(ntuple(j->input[offset+j],Val(8)))
 @inbounds y=fe_storage(ntuple(j->input[offset+8+j],Val(8)))
 x,y
end
function secp_compute(input)
 x,y=load_point(input,Int32(0));z=fe_one();zall=z
 for point in Int32(1):Int32(511)
  bx,by=load_point(input,point*Int32(16));x,y,z=ge_add(x,y,z,bx,by);zall=fe_mul(zall,z)
 end
 fe_bytes(fe_inv(zall))
end
''')
(p/'field_math.jl').write_text('\n'.join(out))
