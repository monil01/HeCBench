"""Independent PRESENT-80 specification oracle, using integer S-box/bit permutation."""
import argparse
import hashlib
import json
from pathlib import Path
SBOX=(12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2)
KATS=("5579c1387b228445","e72c46c0f5945049","a112ffc72f68417b","3333dcd3213210d2")
SOURCE="https://crypto.orange-labs.fr/papers/ches2007-450.pdf"

def encrypt(plain,key):
    state=int.from_bytes(plain,"big")
    register=int.from_bytes(key,"big")
    mask=(1<<80)-1
    for counter in range(1,32):
        state^=register>>16
        state=sum(SBOX[(state>>(4*nibble))&15]<<(4*nibble) for nibble in range(16))
        state=sum(((state>>bit)&1)<<(16*bit%63 if bit<63 else 63) for bit in range(64))
        register=((register<<61)&mask)|(register>>19)
        register=(register&((1<<76)-1))|(SBOX[register>>76]<<76)
        register^=counter<<15
    return (state^(register>>16)).to_bytes(8,"big")

def build_fixture():
    plains=[];keys=[];ciphers=[]
    for block in range(1025):
        plain=bytes((37*block+19*lane+11)%256 for lane in range(8))
        key=bytes((53*block+23*lane+7+(block>>8))%256 for lane in range(10))
        if block<4:
            plain=bytes([255 if block>=2 else 0])*8
            key=bytes([255 if block%2 else 0])*10
        cipher=encrypt(plain,key)
        if block<4:assert cipher.hex()==KATS[block]
        plains.append(plain);keys.append(key);ciphers.append(cipher)
    return b"".join(plains),b"".join(keys),b"".join(ciphers)

def check_result(path):
    plain,key,expected=build_fixture();result=json.loads(Path(path).read_text());rows=[]
    for model,record in result["models"].items():
        for trial in record["trials"]:
            data=Path(trial["output_artifact"]["saved_path"]).read_bytes()
            rows.append({"model":model,"trial":trial["trial"],"all8200ciphertext_bytes_match_independent_spec":data==expected,"four_published_known_answers_match":all(data[8*i:8*i+8].hex()==KATS[i]for i in range(4)),"sha256":hashlib.sha256(data).hexdigest()})
    assert len(rows)==20 and all(row["all8200ciphertext_bytes_match_independent_spec"]and row["four_published_known_answers_match"]for row in rows)
    return {"trials":rows,"known_answers":KATS,"source":SOURCE,"source_location":"AppendixI,page16(physicalPDFindex15)","scope":"AlgorithmconformanceforPRESENT80only,notsecuritycertificationorallkeyspacecoverage"}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--result");args=parser.parse_args()
    print(json.dumps(check_result(args.result)if args.result else {"fixture_sha256":hashlib.sha256(build_fixture()[2]).hexdigest(),"known_answers":KATS,"source":SOURCE},indent=2))
