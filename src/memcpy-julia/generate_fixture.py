from pathlib import Path
import struct,json,hashlib
B=Path(__file__).resolve().parent
n=1<<19
bits=[(i*2654435761+0x9e3779b9)&0xffffffff for i in range(n)]
(B/'pattern_u32.bin').write_bytes(struct.pack('<'+'I'*n,*bits))
(B/'fixture_provenance.json').write_text(json.dumps(dict(count=n,formula='UInt32((zero_based_index*2654435761+0x9e3779b9) mod2^32); actualInt32payloadbitreinterpret',sha256=hashlib.sha256((B/'pattern_u32.bin').read_bytes()).hexdigest(),source_sizes_bytes=[1<<(i+6) for i in range(16)],source_default_repeat=8192,study_repeat=20,patterns=['source_allones','full_signed_Int32_varied_bits'],host_memory='pageable native malloc / JuliaVector, no pinning',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
