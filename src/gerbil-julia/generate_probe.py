from pathlib import Path
import random,struct,json,collections,hashlib
p=Path(__file__).resolve().parent;r=random.Random(314159);seqs=[''.join(r.choice('ACGT')for _ in range(28))for _ in range(37)];tr=str.maketrans('ACGT','TGCA');strings=[min(s,s.translate(tr)[::-1])for s in seqs];keys=[]
for i in range(257):
 s=strings[(i*13+i//37)%37];v=0
 for b in s:v=(v<<2)|'ACGT'.index(b)
 keys.append(v<<8)
counts=collections.Counter(keys);words=[z for x in keys for z in[x&0xffffffff,x>>32]];data=struct.pack('<II',len(keys),len(counts))+struct.pack('<'+'I'*len(words),*words)
for k,n in sorted(counts.items()):data+=struct.pack('<QI',k,n)
(p/'probe.bin').write_bytes(data);(p/'probe_manifest.json').write_text(json.dumps({'k':28,'keys':257,'unique':len(counts),'canonical_reverse_complement':True,'source_layout':'UInt64 numerical basepacking high56bits, low8zero; little-endian twoUInt32words','fixture_sha256':hashlib.sha256(data).hexdigest(),'sequences':strings,'counts':[[str(k),n]for k,n in sorted(counts.items())]},indent=2)+'\n')
