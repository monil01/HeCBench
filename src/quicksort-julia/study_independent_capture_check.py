"""Full-value independent final-output/permutation/guard oracle; no GPU work."""
from pathlib import Path
import array,struct,json,hashlib,sys
P=Path(__file__).resolve().parent
SG=0xdeadbeef
f=array.array('Q',(P/'fixtures_u64.bin').read_bytes());pos=1;suffix=[];details=[]
for _ in range(f[0]):
 tid,n,mode,r=map(int,f[pos:pos+4]);pos+=4;a=f[pos:pos+r];pos+=r;dn=f[pos:pos+r];pos+=r
 fmt=('I','f','d')[tid];size=(4,4,8)[tid];ng=256//size
 dec=lambda x:struct.unpack('<'+fmt,int(x).to_bytes(size,'little'))[0]
 enc=lambda x:int.from_bytes(struct.pack('<'+fmt,x),'little')
 original=list(map(dec,a[:n]));expected=sorted(original);guard=a[n]
 assert all(x==guard for x in a[n:])and all(x==guard for x in dn[n:]);assert list(map(dec,dn[:n]))==expected
 if mode==1:assert original==[42]*n
 else:assert sorted(original)==list(range(1,n+1))
 sorted_bits=[enc(x)for x in expected]+[guard]*(r-n)
 suffix +=sorted_bits+list(dn)+[guard]*(4*ng)+(r-n)*[guard]
 suffix +=2*[SG,SG,guard,SG,SG]+[SG]*10+4*[SG,SG,guard,SG]
 details.append({'type':('UInt32','Float32','Float64')[tid],'n':n,'pattern':mode,'rounded_payload_words':r,'permutation_or_duplicate_count_checked':True})
assert pos==len(f)
golden=f+array.array('Q',suffix);raw=golden.tobytes();sha=hashlib.sha256(raw).hexdigest();rows=[]
for filename in sys.argv[1:]:
 q=Path(filename);got=q.read_bytes();ok=got==raw;rows.append({'path':str(q),'sha256':hashlib.sha256(got).hexdigest(),'exact_full_sorted_bits_permutation_readonly_padding_outer_control_guards':ok});assert ok
print(json.dumps({'golden_sha256':sha,'prefix_words':len(f),'output_words':len(golden),'cases':details,'artifacts':rows},indent=2))
