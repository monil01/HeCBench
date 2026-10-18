
from pathlib import Path
import numpy as np,json,hashlib
p=Path(__file__).resolve().parent;rng=np.random.default_rng(20261018)
lengths=np.r_[np.tile([0,1,2,3,4,5,8,9,16,17,32,33,64,65,128,129,256,257,512,513,1024,1025,2048,2049,4096,5000],12),rng.integers(0,5001,size=64)].astype("i4")
offsets=np.r_[0,np.cumsum(lengths)].astype("<i4");n=int(offsets[-1]);ns=len(lengths)
pairkeys=np.empty(n,dtype="<i4");keyonly=np.empty(n,dtype="<i4")
for seg in range(ns):
 s,e=offsets[seg:seg+2];order=rng.permutation(e-s).astype("i4");pairkeys[s:e]=order-(e-s)//2;keyonly[s:e]=(order%17)-8
vals=(np.uint64(0xfedc000000000000)+np.arange(n,dtype="u8")).astype("<u8")
out1=keyonly.copy();out2=pairkeys.copy();outv=vals.copy()
for s,e in zip(offsets[:-1],offsets[1:]):
 out1[s:e]=np.sort(keyonly[s:e]);order=np.argsort(pairkeys[s:e],kind="stable");out2[s:e]=pairkeys[s:e][order];outv[s:e]=vals[s:e][order]
arrays={"keys_case1_i32.bin":keyonly,"keys_case2_i32.bin":pairkeys,"values_case2_u64.bin":vals,"offsets_i32.bin":offsets,"oracle_case1_i32.bin":out1,"oracle_case2_i32.bin":out2,"oracle_values_case2_u64.bin":outv}
bins=np.minimum(np.array([0 if x<=1 else int(x-1).bit_length()for x in lengths]),12);counts=np.bincount(bins,minlength=13)
for name,a in arrays.items():(p/name).write_bytes(a.tobytes())
meta={"N":n,"segments":ns,"max_segment_length":int(lengths.max()),"all13bins_counts":counts.tolist(),"empty_segments":int(np.sum(lengths==0)),"keys_only_duplicates":True,"pair_keys_unique_within_each_segment":True,"pair_values_full_UInt64_high_bits":True,"source_unstable_ties":"Nativepairtieorderunspecified, unique-per-segmentpairkeysavoidfalsevalueequality; keysonly duplicates retained.","input_scope":"Designedallbins/sizes0..5000/negativekeys deterministicCPUfixture, notoriginal32million-elementRNGprogram.","NumPy":np.__version__,"CPU_generated":True,"fixtures":{name:{"sha256":hashlib.sha256((p/name).read_bytes()).hexdigest(),"bytes":(p/name).stat().st_size,"dtype":str(a.dtype)}for name,a in arrays.items()}}
(p/"study_sizes.txt").write_text(f"{n} {ns}\n")
(p/"fixture_provenance.json").write_text(json.dumps(meta,indent=2)+"\n");print(n,ns,counts)
