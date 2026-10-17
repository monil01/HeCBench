# Regenerate the named AES128 CPU fixture without editing any kernel or reference.
# Requires the local OpenSSL CLI; fixed public test key and plaintext.
from pathlib import Path
import json,hashlib,re,subprocess
p=Path(__file__).resolve().parent
r=p.parent.parent
header=(r/'src/aes-cuda/aes.h').read_text();sbox=bytes(int(x,16)for x in re.findall(r'0x[0-9a-fA-F]+',re.search(r'uchar sbox\[256\]\s*=\s*\{(.*?)\};',header,re.S).group(1)));inverse=bytes(int(x,16)for x in re.findall(r'0x[0-9a-fA-F]+',re.search(r'uchar rsbox\[256\]\s*=\s*\{(.*?)\};',header,re.S).group(1)));assert len(sbox)==len(inverse)==256
key=bytes(range(16));expanded=list(key);rc=1
while len(expanded)<176:
 temp=expanded[-4:]
 if len(expanded)%16==0:
  temp=[sbox[x]for x in temp[1:]+temp[:1]];temp[0]^=rc;rc=((rc<<1)^ (0x11b if rc&128 else 0))&255
 for value in temp:expanded.append(expanded[-16]^value)
def transpose(data):
 return b''.join(bytes(data[o+c*4+row]for row in range(4)for c in range(4))for o in range(0,len(data),16))
plain=bytearray((17*i+13*(i//16)+11*(i//256))%256 for i in range(4096));kat=bytes.fromhex('00112233445566778899aabbccddeeff');plain[:16]=transpose(kat);standard=transpose(plain);cmd=['openssl','enc','-aes-128-ecb','-K',key.hex(),'-nopad','-nosalt'];enc=subprocess.run(cmd,input=standard,capture_output=True,check=True).stdout;assert enc[:16].hex()=='69c4e0d86a7b0430d8cdb78070b4c55a';cipher=transpose(enc);dec=subprocess.run(cmd+['-d'],input=enc,capture_output=True,check=True).stdout;assert dec==standard
fixture={'key':key.hex(),'roundkey':transpose(bytes(expanded)).hex(),'sbox':sbox.hex(),'inverse':inverse.hex(),'plain':plain.hex(),'cipher':cipher.hex()};(p/'study_fixture.txt').write_text(''.join(k+' '+v+'\n'for k,v in fixture.items()));(p/'study_fixture_provenance.json').write_text(json.dumps({'generator':'PythonstandardAES128keyexpansion plus native state/key transpose; OpenSSLCPUAESECBallblocks','openssl_version':subprocess.run(['openssl','version'],capture_output=True,text=True,check=True).stdout.strip(),'command':cmd,'nbytes':4096,'key_hex':key.hex(),'standard_first_plaintext_hex':kat.hex(),'standard_first_ciphertext_hex':enc[:16].hex(),'nist_known_answer_pass':True,'all_blocks_decrypt_roundtrip_pass':True,'native_layout':'native4x4row-majorstate; transposeeachblockbefore/afterstandardAES','fixture_sha256':hashlib.sha256((p/'study_fixture.txt').read_bytes()).hexdigest(),'native_sbox_header_sha256':hashlib.sha256(header.encode()).hexdigest(),'scope':'Kernelonly namedpost-grayscale bytes; no externalbitmapdependency; AES128only, notsecurityclaim'},indent=2)+'\n')

q=p/'study_fixture_provenance.json';d=json.loads(q.read_text());d['known_answer_source']={'title': 'FIPS197(2001),AppendixC.1AES128example, pages35–37', 'url': 'https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf', 'current_standard_url': 'https://csrc.nist.gov/pubs/fips/197/final', 'source_status': 'Historical original archived after2023editorialupdate; NISTstatesnoalgorithmtechnicalchange', 'verified_source_plaintext': '00112233445566778899aabbccddeeff', 'verified_source_key': '000102030405060708090a0b0c0d0e0f', 'verified_source_ciphertext': '69c4e0d86a7b0430d8cdb78070b4c55a'};q.write_text(json.dumps(d,indent=2)+'\n')
