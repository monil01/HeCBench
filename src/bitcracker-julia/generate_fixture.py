"""Known synthetic SHA256/AES-CCM operator fixtures, no external target data."""
from pathlib import Path
import hashlib,json,struct
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
P=Path(__file__).resolve().parent
# Independent standard-library SHA256 and FIPS AES256 ECB implementation checks.
assert hashlib.sha256(b'abc').hexdigest()=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
e=Cipher(algorithms.AES(bytes(range(32))),modes.ECB()).encryptor()
assert (e.update(bytes.fromhex('00112233445566778899aabbccddeeff'))+e.finalize()).hex()=='8ea2b7ca516745bfeafc49904b496089'
salt=bytes(range(16));nonce=bytes(31+i*3 for i in range(12));padding=b'\x80'+bytes(31)+(88*8).to_bytes(8,'big')
passwords=['Bench000','FixedSyntheticNumericFixtureForSHA256PaddingBoundaryABCDZ'[:55]]
assert len(passwords[0])==8 and len(passwords[1])==55
keys=[];rows=[];wire=[3]+list(salt)+list(padding)
for text in passwords:
 first=hashlib.sha256(hashlib.sha256(text.encode('utf-16le')).digest()).digest();key=bytes(32)
 for counter in range(1048576):key=hashlib.sha256(key+first+salt+counter.to_bytes(8,'little')).digest()
 keys.append(key)
for ci in range(3):
 pi=0 if ci==0 else 1;pw=passwords[pi];key=keys[pi];plain=bytes((i*7+19)%256 for i in range(44));cipher=AESCCM(key,tag_length=16).encrypt(nonce,plain,None);vmk=cipher[:44];tag=bytearray(cipher[44:])
 if ci==2:tag[0]^=1
 try:decoded=AESCCM(key,tag_length=16).decrypt(nonce,vmk+tag,None);valid=decoded==plain
 except Exception:valid=False
 assert valid==(ci!=2)
 raw=pw.encode('utf-16le');length=64 if len(pw)<=27 else 128;padded=raw+b'\x80'+bytes(length-len(raw)-9)+(len(raw)*8).to_bytes(8,'big');words=list(struct.unpack('>'+str(length//4)+'I',padded));words+= [0]*(32-len(words))
 if len(pw)<=27:words[14]=0xffffffff
 iv=bytes([2])+nonce+bytes([0,0,1]);maciv=bytes([2])+nonce+bytes(3);computed=bytes([0x3a])+nonce+bytes([0,0,44]);expected=0 if valid else 0xffffffff
 wire+= [len(pw)]+words+list(vmk)+list(iv)+list(tag)+list(maciv)+list(computed)+[expected]
 rows.append({'case':ci,'known_synthetic_password':pw,'length':len(pw),'first_double_SHA256':hashlib.sha256(hashlib.sha256(raw).digest()).hexdigest(),'full_1048576_iteration_KDF_key':key.hex(),'plaintext':plain.hex(),'nonce':nonce.hex(),'ciphertext':vmk.hex(),'tag':tag.hex(),'expected_MAC_valid':valid,'expected_found_initial_minus1_plus_single_valid':expected,'source_password_words':words})
(P/'fixture.bin').write_bytes(struct.pack('<'+str(len(wire))+'I',*wire));(P/'fixture_manifest.json').write_text(json.dumps({'cases':rows,'iterations':1048576,'salt':salt.hex(),'padding':padding.hex(),'wire_uint32_count':len(wire),'fixture_sha256':hashlib.sha256((P/'fixture.bin').read_bytes()).hexdigest(),'contract':'FullsourceKDF1M +AES256CCM44B, knownsyntheticmessagesonly; onecandidateperlaunch, positive8/55charandnegative55chartag, no60kbatch/racecoverage.'},indent=2)+'\n')
(P/'synthetic_original_cli_hash.txt').write_text('$bitlocker$0$16$'+salt.hex()+'$1048576$12$'+nonce.hex()+'$60$'+bytes.fromhex(rows[0]['tag']).hex()+rows[0]['ciphertext']+'\n');(P/'synthetic_original_cli_dictionary.txt').write_text(passwords[0]+'\n')
print('Known syntheticfixture',len(wire),'words',hashlib.sha256((P/'fixture.bin').read_bytes()).hexdigest())
