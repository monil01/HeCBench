from pathlib import Path
import json,hashlib,shutil,struct,ctypes
R=Path('/home/imo/HeCBench_julia');B=R/'src/ntt-julia';N=65536;P=0xffffffff00000001
lib=ctypes.CDLL(None);lib.srand(123);source=[(lib.rand()<<32)|lib.rand()for _ in range(N)]
cases=[source,[(P-1 if i%5==0 else (i*0x9e3779b97f4a7c15)%P)for i in range(N)],[((i//512+1)*37 if i%512 in [0,127,511]else 0)for i in range(N)]]
# Independent direct eight-point finite-field DFT, then explicit source transpose/pair butterfly/projection.
w=pow(2,24,P);assert pow(w,8,P)==1 and pow(w,4,P)==P-1
powers=[[pow(w,j*k,P)for k in range(8)]for j in range(8)];oracle=[]
for inp in cases:
 out=[None]*N
 for b in range(128):
  shared=[0]*512
  for t in range(64):
   base=(b<<9)|((t&0x3e)<<3)|(t&1);v=[inp[base|(i<<1)]for i in range(8)]
   d=[sum(v[k]*powers[j][k]for k in range(8))%P for j in range(8)]
   for i in range(8):shared[(t<<3)|i]=d[i]*pow(2,((t&1)<<2)*i*3,P)%P
  for t in range(64):
   f=((t&0x38)<<3)|(t&7);tm=(b<<9)|f;v=[shared[f|(i<<3)]for i in range(8)]
   s=[z for i in range(4)for z in [(v[2*i]+v[2*i+1])%P,(v[2*i]-v[2*i+1])%P]]
   for i in range(8):
    addr=(((tm|(i<<3))&15)<<12)|((tm|(i<<3))>>4);assert out[addr]is None;out[addr]=(s[i]*18446462594437939201%P)&0xffffffff
 assert None not in out;oracle.extend(out)
(B/'inputs_u64.bin').write_bytes(struct.pack('<'+'Q'*(N*3),*(x for c in cases for x in c)));(B/'oracle_u32.bin').write_bytes(struct.pack('<'+'I'*(N*3),*oracle))
def ev(p):return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
(B/'fixture_provenance.json').write_text(json.dumps(dict(generator=ev(Path('/tmp/draft_ntt.py'))if False else 'generate_fixture_oracle.py',n=N,cases=['actualglibc_srand123rand_uint64','fullcanonicalfield_with_pminus1','three_sparse_impulses_per_sourceblock'],source_scope='Actualsource intt_3_64k_modcrt finalstage only, notcomplete65536-pointinverseNTT/CRT reconstruction',oracle='IndependentPythonBigIntdirect8pointDFT root2^24, twiddle2^shift, explicittranspose/pairDFT, canonical multiply thenUInt32lowwordprojection',assets=[ev(B/'inputs_u64.bin'),ev(B/'oracle_u32.bin')]),indent=2)+'\n')

