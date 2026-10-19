"""CPU-only full schedule/input/guard/AES-CCM oracle, no GPU source helper calls."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
P=Path(__file__).resolve().parent

def audit(path):
 f=list(struct.unpack('<483I',(P/'fixture.bin').read_bytes()));count=1048576;N=count*64;pos=0
 with open(path,'rb')as file,open(P/'schedule_oracle.bin','rb')as gold:
  assert file.seek(0,2)==67112675*4 and gold.seek(0,2)==N*4
  with mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)as wire,mmap.mmap(gold.fileno(),0,access=mmap.ACCESS_READ)as oracle:
   def get(n):
    nonlocal pos
    z=list(struct.unpack_from('<'+str(n)+'I',wire,pos*4));pos+=n;return z
   def guarded(values,byte=False):return [165 if byte else 0xa5a5a5a5]*32+list(values)+[165 if byte else 0xa5a5a5a5]*32
   def same(expected):assert get(len(expected))==expected,'actualinput/readonly/guard mismatch'
   same([3,count]);salt=bytes(f[1:17]);padding=bytes(f[17:57]);same(guarded(salt,True));same(guarded(padding,True));same([0xa5a5a5a5]*32)
   table=pos*4
   for offset in range(0,N*4,8*1024*1024):
    stop=min(offset+8*1024*1024,N*4);assert wire[table+offset:table+stop]==oracle[offset:stop],'fullschedule mismatch'
   pos+=N;same([0xa5a5a5a5]*32);off=57;cases=[];keys={};oracle_labels=[]
   salts=list(struct.unpack('>4I',salt))
   for ci in range(3):
    length=f[off];off+=1;same([length]+salts+salts);inputs=[]
    for size,byte in [(32,False),(44,True),(16,True),(16,True),(16,True),(16,True)]:
     values=f[off:off+size];off+=size;inputs.append(guarded(values,byte));same(inputs[-1])
    expected=f[off];off+=1;same(guarded([0xffffffff]));raw=struct.pack('>32I',*inputs[0][32:-32])[:2*length]
    assert len(raw)==length*2 and all(raw[i]==0 for i in range(1,len(raw),2))
    keyid=(raw,salt)
    if keyid not in keys:
     first=hashlib.sha256(hashlib.sha256(raw).digest()).digest();key=bytes(32)
     for counter in range(count):key=hashlib.sha256(key+first+salt+counter.to_bytes(8,'little')).digest()
     keys[keyid]=key
    vmk=bytes(inputs[1][32:-32]);iv=bytes(inputs[2][32:-32]);mac=bytes(inputs[3][32:-32]);nonce=iv[1:13]
    assert iv==bytes([2])+nonce+bytes([0,0,1]);assert bytes(inputs[4][32:-32])==bytes([2])+nonce+bytes(3);assert bytes(inputs[5][32:-32])==bytes([0x3a])+nonce+bytes([0,0,44])
    try:plain=AESCCM(keys[keyid],tag_length=16).decrypt(nonce,vmk+mac,None);valid=plain==bytes((i*7+19)%256 for i in range(44))
    except Exception:valid=False
    want=0 if valid else 0xffffffff;assert want==expected
    cases.append(inputs+[guarded([want])]);oracle_labels.append(valid)
   assert off==len(f)and pos==67110908
   for buffers in cases:
    for values in buffers:same(values)
   assert pos==67112675
 return {'all_pass':True,'complete_schedule_words':N,'actual_input_prefix_words':67110908,'full_artifact_words':67112675,'full_KDF_iterations_per_case':count,'independent_full_KDF_keys':len(keys),'independent_AESCCM_labels':oracle_labels,'single_candidate_per_kernel':True,'race_or_60kbatch_coverage_claim':False,'integer_output_max_abs_error':0,'scope':'Knownsyntheticpositive8/55char, negativealteredtag55char, fullsource1MiterationKDF/AESCCM andfull1Mschedule; no externaldata ordefaultCLIcertification.'}
if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2))
