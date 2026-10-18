"""Independent coordinate Keccak-f[800] oracle, LFSR constants, published XKCP KATs.
Specification: https://keccak.team/keccak_specs_summary.html
KATs: https://raw.githubusercontent.com/XKCP/XKCP/master/tests/TestVectors/KeccakF-800-IntermediateValues.txt
"""
import array,argparse,json,hashlib
from pathlib import Path
MASK=2**32-1
RHO=(0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14)
def constants():
 rc=[];r=1
 for _ in range(22):
  value=0
  for j in range(7):
   if r&1:value^=1<<((1<<j)-1)
   r=((r<<1)^(0x71 if r&128 else 0))&255
  rc.append(value&MASK)
 return rc
RC=constants()
def rol(v,n):
 n%=32;return((v<<n)|(v>>(32-n)))&MASK

def permutation(a):
 a=list(a)
 for rc in RC:
  c=[a[x]^a[x+5]^a[x+10]^a[x+15]^a[x+20]for x in range(5)]
  delta=[c[(x-1)%5]^rol(c[(x+1)%5],1)for x in range(5)]
  for x in range(5):
   for y in range(5):a[x+5*y]^=delta[x]
  b=[0]*25
  for x in range(5):
   for y in range(5):b[y+5*((2*x+3*y)%5)]=rol(a[x+5*y],RHO[x+5*y])
  a=[b[x+5*y]^((~b[(x+1)%5+5*y])&b[(x+2)%5+5*y])for y in range(5)for x in range(5)];a[0]^=rc
 return a

KAT1=[int(v,16)for v in '''E531D45D F404C6FB 23A0BF99 F1F8452F 51FFD042 E539F578 F00B80A7 AF973664 BF5AF34C 227A2424 88172715 9F685884 B15CD054 1BF4FC0E 6166FA91 1A9E599A A3970A1F AB659687 AFAB8D68 E74B1015 34001A98 4119EFF3 930A0E76 87B28070 11EFE996'''.split()]
KAT2=[int(v,16)for v in '''75BF2D0D 9B610E89 C826AF40 64CD84AB F905BDD6 BC832835 5F8001B9 15662CCE 8E38C95E 701FE543 1B544380 89ACDEFF 51EDB5DE 0E9702D9 6C19AA16 A2913EEE 60754E9A 9819063C F4709254 D09F9084 772DA259 1DB35DF7 5AA60162 358825D5 B3783BAB'''.split()]

def expected(input):
 leaf=[0]*1536
 for bid in range(3):
  for tid in range(64):
   state=[0]*25
   for k in range(3):
    for word in range(8):state[word]^=input[tid+word*64+k*512+bid*1536]
    state=permutation(state)
   for word in range(8):leaf[tid+word*64+bid*512]=state[word]
 root=[0]*25;trace=[]
 for _ in range(8):
  for k in range(192):
   for word in range(8):root[word]^=leaf[k*8+word]
   root=permutation(root)
  trace.extend(root)
 return leaf+trace

def audit(path):
 assert permutation([0]*25)==KAT1;assert permutation(KAT1)==KAT2
 d=json.loads(Path(path).read_text());reports=[];cache={}
 for model,m in d['models'].items():
  for trial in m.get('trials',[]):
   source=Path(trial['output_artifact']['saved_path']);blob=source.read_bytes();digest=hashlib.sha256(blob).hexdigest()
   if digest not in cache:
    a=array.array('I');a.frombytes(blob);assert a.itemsize==4;off=0;inputs=[]
    for case in range(2):
     inp=list(a[off:off+4608]);off+=4608;assert list(a[off:off+22])==RC;off+=22;assert all(x==0 for x in a[off:off+25+1536]);off+=25+1536;assert list(a[off:off+6])==[64,3,3,8,32,32];off+=6
     want=[0]*4608 if case==0 else[((i*2654435761)&MASK)^(i>>3)^0xa5a55a5a for i in range(4608)];assert inp==want;inputs.append(inp)
    assert off==12394;cases=[]
    for case,inp in enumerate(inputs):
     got=list(a[off:off+1736]);off+=1736;want=expected(inp);assert got==want;cases.append({'case':case+1,'all_192_leaf_chaining_words_checked':1536,'all_eight_25_lane_root_states_checked':200,'full_words_exact':True,'final_root_first_eight_lanes_hex':[f'{x:08x}'for x in got[-25:-17]]})
    assert off==len(a)==15866;cache[digest]=cases
   reports.append({'model':model,'trial':trial['trial'],'artifact_sha256':digest,'cases':cache[digest]})
 return {'benchmark':'keccaktreehash','result':str(Path(path).resolve()),'captured_trials_checked':len(reports),'distinct_complete_artifacts_independently_checked':len(cache),'published_permutation_KATs_passed':2,'round_constants':'Independently generated8bitLFSR0x71,64bitpositions then32bittruncation; all22constants exact.','permutation_oracle':'Independent 5x5coordinate theta/rho/pi/chi/iota, not sourcein-place24lanecycle; two published800bitfullstateKATs.','specification':'https://keccak.team/keccak_specs_summary.html','known_answer_vectors':'https://raw.githubusercontent.com/XKCP/XKCP/master/tests/TestVectors/KeccakF-800-IntermediateValues.txt','scope':'Historicalcustomtreeconstruction withsourceSoAleafpacking andcontiguous8wordrootchunks;22round32bitKeccak-f800 primitive. NotSHA3orstandardizedtreehash/domainseparation/securitycertificate.','reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)
